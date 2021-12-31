import cv2
import numpy as np

def display_demo_pattern(demo, ptns):
    # display pattern
    cv2.putText(demo, "Pattern: ", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (255,255,255), 2)
    x_axis = 115
    for ptn in ptns:
        cv2.putText(demo, str(ptn[0]), (x_axis,28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ptn[1], 2)
        x_axis += 25

    return demo

def display_demo_palm(demo, palm):
    # draw simulation for palms
    xmin = (palm[0]) - 15
    ymin = (palm[1]) - 15
    xmax = (palm[0]) + 15
    ymax = (palm[1]) + 15
    cv2.rectangle(demo, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (255,0,0), 2)

    return demo

def display_demo_ball(demo, ball):
    # draw simulation for balls
    traces = np.array(ball["trace"])
    # draw only 8 points, prevent chaos display on simulation
    if len(traces) > 8:
        traces = traces[len(traces)-8:]
    for index,_ in enumerate(traces):
        if index == 0: continue
        thickness = index + 2
        cv2.line(demo, tuple(np.array(traces[index - 1]).astype(int)), tuple(np.array(traces[index]).astype(int)), ball["colors"], thickness)

    return demo
