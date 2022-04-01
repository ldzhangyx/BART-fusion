from dataclasses import dataclass
import os


@dataclass
class LyricsCommentData(object):
    music4all_id: str
    songmeanings_id: str
    lyrics: str
    comment: str

    def get_audio_path(self): # get audio path from id
        self.audio_path = os.path.join("Music4All/music4all/audios",
                            self.music4all_id + '.mp3'
                            )
        return self.audio_path