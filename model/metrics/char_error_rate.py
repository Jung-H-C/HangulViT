def char_level_accuracy(predictions, targets, pad_token = None):
    """
        Char-level accuracy를 계산하는 함수

        :param predictions: 모델의 예측 결과 리스트 (batch_size, max_seq_len)
        :param targets: 실제 라벨 리스트 (batch_size, max_seq_len)
        :param pad_token: 패딩 토큰 (옵션) - 패딩된 토큰은 정확도 계산에서 제외됨
        :return: char-level accuracy
    """
    correct_chars = 0
    total_chars = 0

    for pred, target in zip(predictions, targets):
        for p_char, t_char in zip(pred, target):
            if pad_token is not None and t_char == pad_token:
                continue
            if p_char == t_char:
                correct_chars += 1
            total_chars += 1

    if total_chars == 0:
        return 0.0

    return correct_chars / total_chars