import cv2
import numpy as np

def display_demo_pattern(demo, ptns):
    # font settings
    height, width = demo.shape[:2]
    resolution = height+width
    pos_x = int(resolution * 0.01)
    pos_y = int(resolution * 0.03)
    font_size = 0.5 * (resolution) / 600
    font_scale = int(0.6 * (resolution) // 300)
    font_style = cv2.FONT_HERSHEY_DUPLEX

    # display pattern
    cv2.putText(demo, "Pattern: ", (pos_x,pos_y), font_style, font_size, (255,255,255), font_scale)
    x_axis = (resolution)//8 # pattern result pos
    for ptn in ptns:
        cv2.putText(demo, str(ptn[0]), (int(x_axis),pos_y+int(pos_y*0.06)), font_style, font_size, ptn[1], font_scale)
        x_axis += (resolution)//50 # next pattern result pos

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
        cv2.line(demo, tuple(np.array(traces[index - 1]).astype(int)), tuple(np.array(traces[index]).astype(int)),
                 ball["colors"], thickness)

    return demo
