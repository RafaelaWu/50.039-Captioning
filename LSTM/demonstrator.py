import kivy
import json
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
import os
import argparse

import sample
from sample import Vocabulary

kivy.require('1.10.1')
Window.size = (1000,600)

presentation = Builder.load_file('demonstrator.kv')

class MainScreen(BoxLayout):

    def selected(self, filename):
        self.ids.img.source = filename[0]

        sentences = sample.run(self.ids.img.source, args.encoder_path, args.decoder_path)
        for i in range(len(sentences)):
        	sentences[i] = sentences[i][8:-6]

        self.ids.pred.text = '\n'.join(sentences)
        #self.ids.pred.text = 'guess what I have predicted'

class MyApp(App):
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='encoder-10-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='decoder-10-3000.ckpt', help='path for trained decoder')
    args = parser.parse_args()
    MyApp().run()