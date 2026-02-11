#!/usr/bin/env python3
"""
Jenkins Pipeline Notification Script

Sends pipeline status notifications to Telegram using the existing bot configuration.
Can be called from Jenkins post-build steps to notify about success/failure.

Usage:
    python notify_pipeline.py --status failure --job "LSTM_Forecast" --build 42 --error "Training failed"
    python notify_pipeline.py --status success --job "LSTM_Forecast" --build 42 --duration "5m 30s"
"""

import asyncio
import argparse
import configparser
import sys
from datetime import datetime
from pathlib import Path

import telegram


def get_config():
    """Load Telegram configuration from config.ini"""
    config_path = Path(__file__).parent / 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    
    return {
        'bot_token': config['token']['BOT_TOKEN'],
        'chat_id': config['chat']['CHAT_ID']
    }


async def send_notification(message: str, parse_mode: str = 'HTML'):
    """Send a notification message to Telegram"""
    config = get_config()
    bot = telegram.Bot(token=config['bot_token'])
    
    try:
        await bot.send_message(
            chat_id=config['chat_id'],
            text=message,
            parse_mode=parse_mode
        )
        print("✅ Pipeline notification sent to Telegram")
        return True
    except Exception as e:
        print(f"❌ Failed to send notification: {e}")
        return False


def format_failure_message(job_name: str, build_number: str, stage: str = None,
                           error: str = None, branch: str = None, 
                           build_url: str = None) -> str:
    """Format a failure notification message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    message = f"""🚨 <b>Pipeline Failed</b>
━━━━━━━━━━━━━━━━━━

<b>Job:</b> {job_name}
<b>Build:</b> #{build_number}"""

    if branch:
        message += f"\n<b>Branch:</b> {branch}"
    
    if stage:
        message += f"\n<b>Failed Stage:</b> {stage}"
    
    message += f"\n<b>Time:</b> {timestamp}"
    
    if error:
        # Truncate long error messages
        error_display = error[:500] + "..." if len(error) > 500 else error
        message += f"\n\n<b>Error:</b>\n<code>{error_display}</code>"
    
    if build_url:
        message += f"\n\n🔗 <a href='{build_url}'>View Build Logs</a>"
    
    return message


def format_success_message(job_name: str, build_number: str, 
                           duration: str = None, branch: str = None) -> str:
    """Format a success notification message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    message = f"""✅ <b>Pipeline Succeeded</b>
━━━━━━━━━━━━━━━━━━

<b>Job:</b> {job_name}
<b>Build:</b> #{build_number}"""

    if branch:
        message += f"\n<b>Branch:</b> {branch}"
    
    if duration:
        message += f"\n<b>Duration:</b> {duration}"
    
    message += f"\n<b>Time:</b> {timestamp}"
    
    return message


def format_started_message(job_name: str, build_number: str,
                           branch: str = None, build_url: str = None) -> str:
    """Format a pipeline started notification message"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    message = f"""🚀 <b>Pipeline Started</b>
━━━━━━━━━━━━━━━━━━

<b>Job:</b> {job_name}
<b>Build:</b> #{build_number}"""

    if branch:
        message += f"\n<b>Branch:</b> {branch}"
    
    message += f"\n<b>Time:</b> {timestamp}"
    
    if build_url:
        message += f"\n\n🔗 <a href='{build_url}'>View Build</a>"
    
    return message


def main():
    parser = argparse.ArgumentParser(description='Send Jenkins pipeline notifications to Telegram')
    parser.add_argument('--status', required=True, choices=['success', 'failure', 'started'],
                        help='Pipeline status')
    parser.add_argument('--job', required=True, help='Job name')
    parser.add_argument('--build', required=True, help='Build number')
    parser.add_argument('--stage', help='Failed stage name (for failures)')
    parser.add_argument('--error', help='Error message (for failures)')
    parser.add_argument('--branch', help='Git branch name')
    parser.add_argument('--duration', help='Build duration (for success)')
    parser.add_argument('--url', help='Build URL')
    parser.add_argument('--message', help='Custom message (overrides default formatting)')
    
    args = parser.parse_args()
    
    # Use custom message if provided
    if args.message:
        message = args.message
    elif args.status == 'failure':
        message = format_failure_message(
            job_name=args.job,
            build_number=args.build,
            stage=args.stage,
            error=args.error,
            branch=args.branch,
            build_url=args.url
        )
    elif args.status == 'success':
        message = format_success_message(
            job_name=args.job,
            build_number=args.build,
            duration=args.duration,
            branch=args.branch
        )
    elif args.status == 'started':
        message = format_started_message(
            job_name=args.job,
            build_number=args.build,
            branch=args.branch,
            build_url=args.url
        )
    
    # Send the notification
    success = asyncio.run(send_notification(message))
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
