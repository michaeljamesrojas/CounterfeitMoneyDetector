# import simpleaudio as sa

# wave_obj = sa.WaveObject.from_wave_file("Pfifty.wav")
# play_obj = wave_obj.play()

# play_obj.wait_done()
# print("haha")



# from playsound import playsound
# playsound('Pfifty.wav')


# from pydub import AudioSegment
# from pydub.playback import play

# song = AudioSegment.from_wav("Pfifty.wav")
# play(song)



import winsound
winsound.PlaySound("Pfifty.wav", winsound.SND_FILENAME)  
print("haha")
# add winsound.SND_ASYNC flag if you want to wait for it. 
# like winsound.PlaySound("Wet Hands.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)