from quarto_env.constants import PIECE_FEATURES


def line_has_common_feature(piece_ids):
    """
    Return True if the 4 pieces share at least one common attribute.
    If any cell is empty (-1), return False.
    """
    if -1 in piece_ids:
        return False

    features = [PIECE_FEATURES[piece_id] for piece_id in piece_ids]

    for i in range(4):
        values = [feature[i] for feature in features]
        if all(v == values[0] for v in values):
            return True

    return False


def check_win(board):
    for i in range(4):
        row = list(board[i, :])
        if line_has_common_feature(row):
            return True

    for i in range(4):
        col = list(board[:, i])
        if line_has_common_feature(col):
            return True

    diag1 = [board[i, i] for i in range(4)]
    if line_has_common_feature(diag1):
        return True

    diag2 = [board[i, 3 - i] for i in range(4)]
    if line_has_common_feature(diag2):
        return True

    return False
