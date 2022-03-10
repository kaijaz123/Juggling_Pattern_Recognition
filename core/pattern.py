import numpy as np

def pattern_recognition(ball, image_w, image_h, model):
    if len(ball["trace"]) < 2: return
    print(ball)

    if len(ball["trace"]) == 5 and len(ball["hand_seq"]) == 1:
        return str(2)

    """data proprocessing"""
    # calculate ball vertical distance - y (height)
    h1_y = ball["trace"][0][1]
    h2_y = ball["trace"][-1][1]

    h_y = (h2_y + h1_y) // 2
    b_y = np.amin(ball["trace"],axis = 0)[-1]

    bally_distance = (h_y - b_y) / image_h

    # determine hand level
    sequence = ball["hand_seq"]
    if len(sequence) == 1 or sequence[0] == sequence[1]:
        hand_level = 0
    else:
        hand_level = 1

    # pattern prediction
    predict_item = np.array([[hand_level,bally_distance]])
    print(predict_item)
    y_pred = model.predict(predict_item)
    result = np.argmax(y_pred, axis=-1)

    print(result[0]+2)
    # 0 means 1
    if result[0] == 0:
        return str(result[0]+1)
    # the rest +2 to match the pattern
    return str(result[0]+2)
