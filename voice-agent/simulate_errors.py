#!/usr/bin/env python3
"""
Script to simulate API errors by temporarily modifying the .env file
Usage: python simulate_errors.py [stt|llm|tts|all|restore]
"""

import sys
import os
import shutil
from datetime import datetime

def backup_env():
    """Create a backup of the .env file"""
    if os.path.exists('.env'):
        shutil.copy('.env', '.env.backup')
        print("✅ Created backup of .env file")

def restore_env():
    """Restore the .env file from backup"""
    if os.path.exists('.env.backup'):
        shutil.copy('.env.backup', '.env')
        print("✅ Restored .env file from backup")
    else:
        print("❌ No backup file found")

def simulate_error(error_type):
    """Simulate an error by commenting out API keys"""
    backup_env()
    
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return
    
    with open('.env', 'r') as f:
        lines = f.readlines()
    
    modified_lines = []
    for line in lines:
        if error_type == 'stt' and line.startswith('ASSEMBLYAI_API_KEY='):
            modified_lines.append(f"# SIMULATED ERROR - {line}")
        elif error_type == 'llm' and line.startswith('GEMINI_API_KEY='):
            modified_lines.append(f"# SIMULATED ERROR - {line}")
        elif error_type == 'tts' and line.startswith('MURF_API_KEY='):
            modified_lines.append(f"# SIMULATED ERROR - {line}")
        elif error_type == 'all' and any(line.startswith(key) for key in ['ASSEMBLYAI_API_KEY=', 'GEMINI_API_KEY=', 'MURF_API_KEY=']):
            modified_lines.append(f"# SIMULATED ERROR - {line}")
        else:
            modified_lines.append(line)
    
    with open('.env', 'w') as f:
        f.writelines(modified_lines)
    
    print(f"✅ Simulated {error_type.upper()} error - restart your server to test")
    print("💡 Use 'python simulate_errors.py restore' to fix")

def main():
    if len(sys.argv) != 2:
        print("Usage: python simulate_errors.py [stt|llm|tts|all|restore]")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'restore':
        restore_env()
    elif command in ['stt', 'llm', 'tts', 'all']:
        simulate_error(command)
    else:
        print("Invalid command. Use: stt, llm, tts, all, or restore")

if __name__ == "__main__":
    main()