import yt_dlp
import whisper
import os
from pathlib import Path

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': 'audio/%(title)s.%(ext)s'
    }
 
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
 
# Example usage
url="https://www.youtube.com/watch?v=kPa7bsKwL-c"
download_audio(url)
model = whisper.load_model("base")
 
files = os.listdir("audio")
for f in files:
    result = model.transcribe("audio/"+f)
    print(result['text'])
    # Create a file and write content in one line
    output = "txt/"+f+".txt"
    Path(output).write_text(result['text'])
    print("File saved to txt folder!")
    os.remove("audio/"+f)

