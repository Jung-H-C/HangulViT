import torch

class Vocabulary:
    def __init__(self):
        self.vocabulary = ['ㅎ', 'ㅔ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ',
                           'ㅕ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ',
                           'ㅖ', 'ㄱ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ', 'ㅏ', 'ㅐ',
                           'ㅃ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅝ', 'ㅞ', 'ㄸ', 'ㅜ', 'ㅉ', '<sos>', '<eos>', '<pad>']  # len = 54

        self.jamo_to_index = {word: i for i, word in enumerate(self.vocabulary)}

    def process(self, label):
        label_indices = [self.jamo_to_index[char] for char in label]
        print('a')

        input_label = [self.jamo_to_index['<sos>']] + label_indices
        print('b')
        input_label = torch.tensor(input_label, dtype=torch.long)
        print('c')

        output_label = label_indices + [self.jamo_to_index['<eos>']]
        print('d')
        output_label = torch.tensor(output_label, dtype=torch.long)
        print('e')

        return input_label, output_label