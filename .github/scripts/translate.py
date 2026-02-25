#!/usr/bin/env python3

import os
import json
import re
import time
from datetime import datetime
from typing import List, Dict
import requests

# Configuration
CONFIG = {
    'anthropic_api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
    'api_base_url': os.environ.get('ANTHROPIC_API_BASE_URL', 'https://api.anthropic.com'),
    'model': os.environ.get('ANTHROPIC_MODEL', 'claude-3-5-haiku-latest'),
    'model_fallbacks': [
        'claude-3-5-haiku-latest'
    ],
    'languages': {
        'de': 'German',
        'fr': 'French',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'th': 'Thai',
        'ms': 'Malay'
    },
    'source_file': 'AutotranslationTest/en.lproj/Localizable.strings',
    'truth_file': '.github/localization-keys.json',
    'max_retries': 3,
    'retry_delay': 1
}

class LocalizationEntry:
    def __init__(self, key: str, value: str, context: str = ''):
        self.key = key
        self.value = value
        self.context = context or 'No context provided'

    def __repr__(self):
        return f"LocalizationEntry(key={self.key}, value={self.value[:30]}...)"


class ClaudeAPIError(Exception):
    def __init__(self, message: str, status_code: int = None, retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def sanitize_strings_value(text: str) -> str:
    if text is None:
        return ''

    value = str(text).strip()

    quote_pairs = [
        ('"', '"'),
        ('“', '”'),
        ('„', '“'),
        ('«', '»'),
        ('‹', '›')
    ]
    for left, right in quote_pairs:
        if len(value) >= 2 and value.startswith(left) and value.endswith(right):
            value = value[1:-1].strip()
            break

    value = value.replace('\\', '\\\\')
    value = value.replace('"', '\\"')
    value = value.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\\n')
    return value

def upsert_localization_entries(lang_code: str, entries: List[Dict]):
    file_path = f"AutotranslationTest/{lang_code}.lproj/Localizable.strings"
    language_name = CONFIG['languages'][lang_code]

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    else:
        lines = [
            "/*",
            "  Localizable.strings",
            f"  AutotranslationTest - {language_name}",
            "*/",
            ""
        ]

    # Map key -> line index of the key/value line.
    key_line_index: Dict[str, int] = {}
    kv_re = re.compile(r'^\s*"([^"]+)"\s*=\s*"(.*)";\s*$')
    for idx, line in enumerate(lines):
        m = kv_re.match(line)
        if m:
            key_line_index[m.group(1)] = idx
    to_append: List[Dict] = []
    for entry in entries:
        key = entry['key']
        translation = sanitize_strings_value(entry.get('translation', ''))

        if key in key_line_index:
            idx = key_line_index[key]
            lines[idx] = f"\"{key}\" = \"{translation}\";"
        else:
            to_append.append({'key': key, 'translation': translation})

    if to_append:
        if lines and lines[-1].strip() != "":
            lines.append("")
        for entry in to_append:
            lines.append(f"\"{entry['key']}\" = \"{entry['translation']}\";")
            lines.append("")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"Updated {lang_code}.lproj/Localizable.strings")


def parse_strings_file(file_path: str) -> List[LocalizationEntry]:
    """Parse .strings file to extract keys, values, and contexts"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    entries = []
    current_context = ''

    for line in lines:
        line_stripped = line.strip()

        # Extract context comment (single line)
        if line_stripped.startswith('/*') and line_stripped.endswith('*/'):
            current_context = line_stripped[2:-2].strip()
            continue

        # Multi-line comment start
        if line_stripped.startswith('/*') and not line_stripped.endswith('*/'):
            current_context = line_stripped[2:].strip()
            continue

        # Multi-line comment end
        if line_stripped.endswith('*/') and not line_stripped.startswith('/*'):
            current_context += ' ' + line_stripped[:-2].strip()
            continue

        # Multi-line comment middle
        if current_context and not line_stripped.startswith('"'):
            current_context += ' ' + line_stripped
            continue

        # Extract key-value pair
        match = re.match(r'^"([^"]+)"\s*=\s*"(.+)";$', line_stripped)
        if match:
            key, value = match.groups()
            entries.append(LocalizationEntry(key, value, current_context))
            current_context = ''

    return entries


def load_truth_file() -> Dict:
    with open(CONFIG['truth_file'], 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_changes(current_entries: List[LocalizationEntry], truth_data: Dict) -> Dict[str, List]:
    changes = {
        'new': [],
        'modified': [],
        'deleted': []
    }

    current_keys = {entry.key: entry for entry in current_entries}
    truth_keys = set(truth_data['keys'].keys())

    # Find new and modified keys
    for entry in current_entries:
        if entry.key not in truth_keys:
            changes['new'].append(entry)
        elif truth_data['keys'][entry.key] != entry.value:
            changes['modified'].append(entry)

    # Find deleted keys
    for key in truth_keys:
        if key not in current_keys:
            changes['deleted'].append(key)

    return changes


def build_api_headers(api_key: str) -> Dict[str, str]:
    return {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }


def parse_api_error(response: requests.Response) -> ClaudeAPIError:
    status_code = response.status_code
    request_id = response.headers.get('request-id', '')

    error_type = 'unknown_error'
    error_message = response.text.strip()
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error_obj = payload.get('error', {})
            error_type = error_obj.get('type', error_type)
            error_message = error_obj.get('message', error_message)
    except Exception:
        pass

    retryable = status_code >= 500 or status_code == 429
    message = f"HTTP {status_code} ({error_type}): {error_message}"
    if request_id:
        message += f" [request-id: {request_id}]"

    return ClaudeAPIError(message=message, status_code=status_code, retryable=retryable)


def list_available_models(api_key: str) -> List[str]:
    """Fetch available model ids from Anthropic API."""
    url = f"{CONFIG['api_base_url'].rstrip('/')}/v1/models"
    response = requests.get(url, headers=build_api_headers(api_key), timeout=30)

    if not response.ok:
        raise parse_api_error(response)

    data = response.json()
    if not isinstance(data, dict):
        return []

    models = data.get('data', [])
    if not isinstance(models, list):
        return []

    model_ids = []
    for model in models:
        if isinstance(model, dict):
            model_id = model.get('id')
            if isinstance(model_id, str) and model_id:
                model_ids.append(model_id)

    return model_ids


def resolve_model(api_key: str) -> str:
    requested = CONFIG['model']

    try:
        available = list_available_models(api_key)
    except Exception as error:
        print(f"Could not fetch model list. Continuing with configured model '{requested}'.")
        print(f"   Reason: {error}")
        return requested

    if not available:
        print(f"Model list was empty. Continuing with configured model '{requested}'.")
        return requested

    if requested in available:
        return requested

    for fallback in CONFIG['model_fallbacks']:
        if fallback in available:
            print(f"Configured model '{requested}' is unavailable. Using '{fallback}' instead.")
            return fallback

    auto_selected = next((m for m in available if 'haiku' in m.lower()), available[0])
    print(f"Configured model '{requested}' is unavailable. Using first available model '{auto_selected}'.")
    return auto_selected


def translate_with_claude(text: str, context: str, target_language: str, api_key: str, model: str) -> str:
    """Call Claude Messages API for translation."""
    prompt = f"""You are a professional translator for mobile applications.

Context: {context}
Source language: English
Target language: {target_language}

Translate the following text to {target_language}. Important guidelines:
- Maintain the same tone and formality level
- Keep the approximate length similar to the original
- Preserve any placeholders like %@, %d, {{0}}, etc. exactly as they appear
- Consider the context to provide the most appropriate translation
- Return ONLY the translated text, no explanations or additional comments

Text to translate: "{text}"

Translation:"""
    url = f"{CONFIG['api_base_url'].rstrip('/')}/v1/messages"

    data = {
        'model': model,
        'max_tokens': 1024,
        'messages': [{
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': prompt
            }]
        }]
    }

    response = requests.post(
        url,
        headers=build_api_headers(api_key),
        json=data,
        timeout=30
    )

    if not response.ok:
        raise parse_api_error(response)

    result = response.json()

    content_blocks = result.get('content', [])
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and block.get('type') == 'text' and 'text' in block:
                return block['text'].strip()

    raise ValueError('Invalid response format from Claude API')


def translate_with_retry(text: str, context: str, target_language: str, api_key: str, model: str) -> str:
    for attempt in range(CONFIG['max_retries']):
        try:
            translation = translate_with_claude(text, context, target_language, api_key, model)
            return translation
        except ClaudeAPIError as error:
            print(f"   Translation attempt {attempt + 1} failed for {target_language}: {str(error)}")
            if not error.retryable:
                raise
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(CONFIG['retry_delay'] * (attempt + 1))
        except Exception as error:
            print(f"   Translation attempt {attempt + 1} failed for {target_language}: {str(error)}")
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(CONFIG['retry_delay'] * (attempt + 1))

    raise Exception(f"Failed to translate after {CONFIG['max_retries']} attempts")


def update_localization_file(lang_code: str, entries: List[Dict]):
    # Backward-compatible wrapper.
    upsert_localization_entries(lang_code, entries)


def remove_deleted_keys(lang_code: str, deleted_keys: List[str]):
    file_path = f"AutotranslationTest/{lang_code}.lproj/Localizable.strings"

    if not os.path.exists(file_path):
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    filtered_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line contains a deleted key
        match = re.match(r'^"([^"]+)"\s*=\s*"(.+)";$', line.strip())
        if match and match.group(1) in deleted_keys:
            # Remove previous comment line if exists
            if filtered_lines and filtered_lines[-1].strip().startswith('/*'):
                filtered_lines.pop()
            i += 1
            continue

        filtered_lines.append(line)
        i += 1

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered_lines))

    print(f"Removed {len(deleted_keys)} deleted keys from {lang_code}.lproj/Localizable.strings")


def update_truth_file(current_entries: List[LocalizationEntry]):
    truth_data = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'keys': {entry.key: entry.value for entry in current_entries}
    }

    with open(CONFIG['truth_file'], 'w', encoding='utf-8') as f:
        json.dump(truth_data, f, indent=2, ensure_ascii=False)

    print("Updated source of truth file")


def main():
    print("Starting auto-translation process...\n")

    # Validate API key
    api_key = CONFIG['anthropic_api_key']
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set")
        exit(1)

    print("Resolving available Claude model...")
    resolved_model = resolve_model(api_key)
    print(f"   Using model: {resolved_model}\n")

    # Parse current English file
    print("Parsing English localization file...")
    current_entries = parse_strings_file(CONFIG['source_file'])
    print(f"   Found {len(current_entries)} entries\n")

    # Load source of truth
    print("Loading source of truth...")
    truth_data = load_truth_file()
    print(f"   Previous state: {len(truth_data['keys'])} keys\n")

    # Detect changes
    print("Detecting changes...")
    changes = detect_changes(current_entries, truth_data)
    print(f"   New keys: {len(changes['new'])}")
    print(f"   Modified keys: {len(changes['modified'])}")
    print(f"   Deleted keys: {len(changes['deleted'])}\n")

    if not changes['new'] and not changes['modified'] and not changes['deleted']:
        print("No changes detected. Nothing to do!")
        exit(0)

    # Prepare entries to translate
    entries_to_translate = changes['new'] + changes['modified']

    if entries_to_translate:
        print(f"Translating {len(entries_to_translate)} entries to {len(CONFIG['languages'])} languages...\n")

        # Translate for each language
        for lang_code, lang_name in CONFIG['languages'].items():
            print(f"\nProcessing {lang_name} ({lang_code})...")

            translations = []
            fatal_api_error = None

            for entry in entries_to_translate:
                try:
                    print(f"   Translating: {entry.key}")
                    translation = translate_with_retry(entry.value, entry.context, lang_name, api_key, resolved_model)
                    translations.append({
                        'key': entry.key,
                        'translation': translation
                    })

                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                except ClaudeAPIError as error:
                    print(f"   Failed to translate {entry.key}: {str(error)}")
                    if not error.retryable:
                        fatal_api_error = error
                        break

                    translations.append({
                        'key': entry.key,
                        'translation': f"TODO: Translation failed - {entry.value}",
                    })
                except Exception as error:
                    print(f"   Failed to translate {entry.key}: {str(error)}")
                    translations.append({
                        'key': entry.key,
                        'translation': f"TODO: Translation failed - {entry.value}",
                    })

            if fatal_api_error is not None:
                print(f"   {lang_name}: non-retryable API error, stopping run.")
                print("\nAborting translation. Fix the API/model configuration and rerun.")
                exit(1)

            # Update only the keys we translated/changed; keep existing file order.
            update_localization_file(lang_code, translations)

    # Handle deleted keys
    if changes['deleted']:
        print(f"\nRemoving {len(changes['deleted'])} deleted keys from all languages...\n")

        for lang_code in CONFIG['languages'].keys():
            remove_deleted_keys(lang_code, changes['deleted'])

    # Update source of truth
    print("\nUpdating source of truth file...")
    update_truth_file(current_entries)

    print("\nAuto-translation completed successfully!")

    # Output summary
    summary = {
        'new_keys': len(changes['new']),
        'modified_keys': len(changes['modified']),
        'deleted_keys': len(changes['deleted']),
        'languages_updated': len(CONFIG['languages'])
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
