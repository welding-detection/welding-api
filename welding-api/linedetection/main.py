import os
import cv2
import roi_click
import p1

filepath = "../data/welding_data/origin_image/"
filelist = os.listdir(filepath)



for i, filename in enumerate(filelist):
    img = cv2.imread(filepath + filename)
    wsf = p1.WorkspaceFinder(img, 1280, 720)

    print(wsf.getData())

    if wsf.send_data['error_messege'] == 200:
        cv2.imshow("workspace", wsf.send_data['img_workspace'])
        cv2.imshow("msk", wsf.send_data['weldingline_mask'])
        cv2.waitKey(0)



cv2.destroyAllWindows()
