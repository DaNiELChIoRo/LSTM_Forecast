import telegram
import asyncio
import os
import configparser

async def send_telegram(message):
    config = configparser.ConfigParser()
    config.read('config.ini')

    try:
        # Telegram Bot Token (Replace with your own bot token)
        BOT_TOKEN = config['token']['BOT_TOKEN']

        # Chat ID (Replace with your own chat ID)
        CHAT_ID = config['chat']['CHAT_ID']

        # Initialize the Telegram bot
        bot = telegram.Bot(token=BOT_TOKEN)

        await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='HTML')
        print("Telegram message sent successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return True

 async def send_image_to_telegram(image_path, caption=None):
            """
            Sends an image to a Telegram chat.

            Args:
                bot_token: The API token of your Telegram bot.
                chat_id: The ID of the chat to send the image to.
                image_path: The file path of the image to send.
                caption: (Optional) A caption to include with the image.
            """
            config = configparser.ConfigParser()
            config.read('config.ini')

            bot_token = config['token']['BOT_TOKEN']
            chat_id = config['chat']['CHAT_ID']
            # chat_id = '955478477'
            try:
                bot = telegram.Bot(token=bot_token)
                async with bot:
                    if caption:
                        await bot.send_photo(chat_id=chat_id, photo=open(image_path, 'rb'), caption=caption)
                    else:
                        await bot.send_photo(chat_id=chat_id, photo=open(image_path, 'rb'))
                print("Image sent successfully!")

            except Exception as e:
                print(f"Error sending image: {e}")