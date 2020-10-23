## speech-to-senti-report
- As an Intern I wrote this script that generates a PDF report based on sentiment analysis, word cloud, 
most used words, personality insights and tone analysis from an input video.
- This script is now in production.
## Demo
[This](https://youtu.be/i6O98o2FRHw) youtube video would generate [this](https://drive.google.com/file/d/1E8oASjydDW1qEXKtA_uh2zO3VIzCc5Ld/view?usp=sharing) PDF report.
## Built With
- [Python](https://www.python.org/)
- [FFMPEG](https://ffmpeg.org/)
- [IBM Watson API's](https://www.ibm.com/in-en/watson/products-services)
- [VADER](https://github.com/cjhutto/vaderSentiment)
- [reportlab](https://pypi.org/project/reportlab/)
## Setup:
- Download [FFMPEG](https://ffmpeg.org/) from [here](https://ffmpeg.org/download.html) and add to PATH for Windows.
- Inside project directory create virtual environment.
```
python -m venv env
.\env\Scripts\Activate
``` 
- Install dependencies
```
pip install -r requirements.txt
```
### Script Walk-through
- Will constantly look for changes within the input directory.
- When a new file gets completely copied/downloaded the script will proceed to split audio from the video file using 
[FFMPEG](https://ffmpeg.org/) and then split the audio file in 4 equal parts to later perform sentiment analysis across 
the duration of the video.
- Following which it uses [Watson Speech-to-text](https://cloud.ibm.com/catalog/services/speech-to-text) to get a 
transcript from the split audio files.
- From the received text it then uses [VADER](https://github.com/cjhutto/vaderSentiment) to perform sentiment analysis.
-  Then generates top used words after performing [lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) 
by using [numpy](https://numpy.org/) and [nltk](https://www.nltk.org/).
- It then gets a predicted personality report of the person speaking in the video. 
- It then gets predicted emotions during the course of the video.
## Usage
- With the input video directory and output report directories are created and specified, the script looks for changes
within input directory and then places the report generated in the output directory. 
## Running the script:
- The script takes in 2 input arguments the input directory and output directory.
```
python main.py \path\to\video\directory \path\to\output\report\directory
```
- As there isn't a platform independent method to determine whether a file has been completely downloaded/copied, 
the condition to determine it was written specifically for Windows as it was deployed on a Windows Server.