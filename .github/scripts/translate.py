#!/usr/bin/env python3

import os
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Tuple
import requests

# Configuration
CONFIG = {
    'anthropic_api_key': os.environ.get('ANTHROPIC_API_KEY', ''),
    'model': 'claude-3-5-haiku-20241022',
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
    """Load source of truth file"""
    with open(CONFIG['truth_file'], 'r', encoding='utf-8') as f:
        return json.load(f)


def detect_changes(current_entries: List[LocalizationEntry], truth_data: Dict) -> Dict[str, List]:
    """Detect changes between current and truth"""
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


def translate_with_claude(text: str, context: str, target_language: str, api_key: str) -> str:
    """Call Claude API for translation"""
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

    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }

    data = {
        'model': CONFIG['model'],
        'max_tokens': 1024,
        'messages': [{
            'role': 'user',
            'content': prompt
        }]
    }

    response = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers=headers,
        json=data,
        timeout=30
    )

    response.raise_for_status()
    result = response.json()

    if 'content' in result and len(result['content']) > 0:
        return result['content'][0]['text'].strip()
    else:
        raise ValueError('Invalid response format from Claude API')


def translate_with_retry(text: str, context: str, target_language: str, api_key: str) -> str:
    """Translate with retry logic"""
    for attempt in range(CONFIG['max_retries']):
        try:
            translation = translate_with_claude(text, context, target_language, api_key)
            return translation
        except Exception as error:
            print(f"   ❌ Translation attempt {attempt + 1} failed for {target_language}: {str(error)}")
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(CONFIG['retry_delay'] * (attempt + 1))

    raise Exception(f"Failed to translate after {CONFIG['max_retries']} attempts")


def update_localization_file(lang_code: str, entries: List[Dict]):
    """Update localization file for a language"""
    file_path = f"AutotranslationTest/{lang_code}.lproj/Localizable.strings"

    language_name = CONFIG['languages'][lang_code]
    content = f"/*\n  Localizable.strings\n  AutotranslationTest - {language_name}\n*/\n\n"

    for entry in entries:
        content += f"/* {entry['context']} */\n"
        content += f'"{entry["key"]}" = "{entry["translation"]}";\n\n'

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Updated {lang_code}.lproj/Localizable.strings")


def remove_deleted_keys(lang_code: str, deleted_keys: List[str]):
    """Remove deleted keys from language file"""
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

    print(f"✓ Removed {len(deleted_keys)} deleted keys from {lang_code}.lproj/Localizable.strings")


def update_truth_file(current_entries: List[LocalizationEntry]):
    """Update source of truth file"""
    truth_data = {
        'last_updated': datetime.utcnow().isoformat() + 'Z',
        'keys': {entry.key: entry.value for entry in current_entries}
    }

    with open(CONFIG['truth_file'], 'w', encoding='utf-8') as f:
        json.dump(truth_data, f, indent=2, ensure_ascii=False)

    print("✓ Updated source of truth file")


def main():
    """Main execution"""
    print("🚀 Starting auto-translation process...\n")

    # Validate API key
    api_key = CONFIG['anthropic_api_key']
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY environment variable is not set")
        exit(1)

    # Parse current English file
    print("📖 Parsing English localization file...")
    current_entries = parse_strings_file(CONFIG['source_file'])
    print(f"   Found {len(current_entries)} entries\n")

    # Load source of truth
    print("📋 Loading source of truth...")
    truth_data = load_truth_file()
    print(f"   Previous state: {len(truth_data['keys'])} keys\n")

    # Detect changes
    print("🔍 Detecting changes...")
    changes = detect_changes(current_entries, truth_data)
    print(f"   New keys: {len(changes['new'])}")
    print(f"   Modified keys: {len(changes['modified'])}")
    print(f"   Deleted keys: {len(changes['deleted'])}\n")

    if not changes['new'] and not changes['modified'] and not changes['deleted']:
        print("✨ No changes detected. Nothing to do!")
        exit(0)

    # Prepare entries to translate
    entries_to_translate = changes['new'] + changes['modified']

    if entries_to_translate:
        print(f"🌍 Translating {len(entries_to_translate)} entries to {len(CONFIG['languages'])} languages...\n")

        # Translate for each language
        for lang_code, lang_name in CONFIG['languages'].items():
            print(f"\n📝 Processing {lang_name} ({lang_code})...")

            translations = []

            for entry in entries_to_translate:
                try:
                    print(f"   Translating: {entry.key}")
                    translation = translate_with_retry(entry.value, entry.context, lang_name, api_key)
                    translations.append({
                        'key': entry.key,
                        'translation': translation,
                        'context': entry.context
                    })

                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                except Exception as error:
                    print(f"   ❌ Failed to translate {entry.key}: {str(error)}")
                    translations.append({
                        'key': entry.key,
                        'translation': f"TODO: Translation failed - {entry.value}",
                        'context': entry.context
                    })

            # Load existing translations
            existing_path = f"AutotranslationTest/{lang_code}.lproj/Localizable.strings"
            existing_entries = []
            if os.path.exists(existing_path):
                try:
                    existing_entries = parse_strings_file(existing_path)
                except Exception:
                    pass

            # Merge with new translations
            existing_keys = {t['key'] for t in translations}
            merged_entries = translations + [
                {
                    'key': e.key,
                    'translation': e.value,
                    'context': e.context
                }
                for e in existing_entries if e.key not in existing_keys
            ]

            update_localization_file(lang_code, merged_entries)

    # Handle deleted keys
    if changes['deleted']:
        print(f"\n🗑️  Removing {len(changes['deleted'])} deleted keys from all languages...\n")

        for lang_code in CONFIG['languages'].keys():
            remove_deleted_keys(lang_code, changes['deleted'])

    # Update source of truth
    print("\n💾 Updating source of truth file...")
    update_truth_file(current_entries)

    print("\n✅ Auto-translation completed successfully!")

    # Output summary
    summary = {
        'new_keys': len(changes['new']),
        'modified_keys': len(changes['modified']),
        'deleted_keys': len(changes['deleted']),
        'languages_updated': len(CONFIG['languages'])
    }

    print("\n📊 Summary:")
    print(json.dumps(summary, indent=2))

    # Write summary for GitHub Actions
    with open('translation-summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
