import librosa
import yaml


class LibriSpeechSoundObject:
    def __init__(self, data_set_path, data_set_name, speaker_id, chapter_number, utterance_number, file_ext='flac'):
        self.data_set_path = data_set_path
        self.data_set_name = data_set_name
        self.speaker_id = speaker_id
        self.chapter_number = chapter_number
        self.utterance_number = utterance_number
        self.file_ext = file_ext

    def params2path(self):
        file_name = f'{self.speaker_id}-{self.chapter_number}-{self.utterance_number}.{self.file_ext}'
        file_path = rf'{self.data_set_path}\{self.data_set_name}\{self.speaker_id}\{self.chapter_number}\{file_name}'
        return file_path

    def read_file(self, fs):
        path = self.params2path()
        speech, fs_file = librosa.load(
            path,
            sr=fs,
            mono=True
        )
        return speech, fs_file


def parse_librispeech_filename(file_name):
    name = file_name.replace('.flac', '')
    speaker_id, chapter_number, utterance_number = name.split('-')
    return speaker_id, chapter_number, utterance_number


def create_librispeech_objects(file_name_list, data_set_path, data_set_name):
    objects = []

    for file_name in file_name_list:
        speaker_id, chapter_number, utterance_number = parse_librispeech_filename(file_name)

        obj = LibriSpeechSoundObject(
            data_set_path=data_set_path,
            data_set_name=data_set_name,
            speaker_id=speaker_id,
            chapter_number=chapter_number,
            utterance_number=utterance_number
        )

        objects.append(obj)

    return objects


def load_librispeech_objects_from_yaml(
        yaml_path,
        data_set_path,
        data_set_name
):
    """
    Loads a YAML file and returns two lists:
    - signal_objects
    - interferer_objects
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    signal_files = config['signal_file_name']
    interferer_files = config['interferer_file_name']

    signal_objects = create_librispeech_objects(signal_files, data_set_path, data_set_name)
    interferer_objects = create_librispeech_objects(interferer_files, data_set_path, data_set_name)

    return signal_objects, interferer_objects
