import cv2
import numpy as np
from ultralytics import YOLO


# mask = np.array([[0, 1, 0],
#                  [1, 1, 1],
#                  [0, 1, 0]]).astype('uint8')


# 교점 찾기 알고리즘
# 출처: https://crazyj.tistory.com/139
def get_crosspt(x11, y11, x12, y12, x21, y21, x22, y22):
    if x12 == x11 and x22 == x21:
        # print("parallel 2")
        return None
    if x12 == x11 or x22 == x21:
        # print('delta x=0')
        if x12 == x11:
            cx = x12
            m2 = (y22 - y21) / (x22 - x21)
            cy = m2 * (cx - x21) + y21
            return cx, cy
        if x22 == x21:
            cx = x22
            m1 = (y12 - y11) / (x12 - x11)
            cy = m1 * (cx - x11) + y11
            return cx, cy

    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1 == m2:
        # print('parallel')
        return None
    # print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return cx, cy


def find_Workspace(img, x_size, y_size):
    '''
        function name: find_workspace(img)
        input: RGB image
        output: RGB image, only workspace
        description:
            용접 시편의 RGB 이미지를 받아와
            Hough 연산을 통해 테두리를 찾고
            이 테두리들의 교점을 활용해 시편의 네 꼭지점을 찾는다.
            마지막에 네 꼭지점을 이용해 원근 투영 변환한 RGB 이미지를 반환한다.

            *** 이 함수는 Workspace를 찾을 만한 선분의 교점이 없을 시에 Error를 일으킵니다. ***
        '''
    img = cv2.resize(img, (x_size, y_size))
    img1 = img.copy()

    img_cp = img.copy()

    # Tuning Point 1
    dst1 = cv2.GaussianBlur(img1, (0, 0), 1)
    canny1 = cv2.Canny(dst1, 100, 150)

    # threshold함수 적용
    _, threshX = cv2.threshold(canny1, 150, 255, cv2.THRESH_BINARY)

    # 직선 검출 및 원본에 선 그리기
    # Tuning Point 2
    lines = cv2.HoughLines(threshX, rho=1, theta=np.pi / 180.0, threshold=150)
    real_lines = []
    for line in lines:
        rho, theta = line[0]
        c = np.cos(theta)
        s = np.sin(theta)
        x0 = c * rho
        y0 = s * rho
        x1 = int(x0 + 1000 * (-s))
        y1 = int(y0 + 1000 * (c))
        x2 = int(x0 - 1000 * (-s))
        y2 = int(y0 - 1000 * (c))
        real_lines.append([(x1, y1), (x2, y2)])
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 교점 찾아서 체크
    crosspts = []
    for i, xys in enumerate(real_lines):
        for j in range(len(real_lines) - i - 1):
            crosspt = get_crosspt(xys[0][0], xys[0][1],
                                  xys[1][0], xys[1][1],
                                  real_lines[i + j + 1][0][0], real_lines[i + j + 1][0][1],
                                  real_lines[i + j + 1][1][0], real_lines[i + j + 1][1][1])
            if crosspt is not None:
                x = int(crosspt[0])
                y = int(crosspt[1])
                # cv2.circle(img, (x, y), 5, (255, 0, 0), 3)
                crosspts.append((x, y))

    # 정리해볼까
    for i, xy in enumerate(crosspts):
        if xy[0] < -2 or xy[0] >= x_size:
            crosspts[i] = None
        if xy[1] < -2 or xy[1] >= y_size:
            crosspts[i] = None

    new_crosspts = [i for i in crosspts if i != None]
    new_crosspts.sort()
    # for xy in new_crosspts:
    #     cv2.circle(img, (xy[0], xy[1]), 5, (255, 0, 0), 3)

    # 중앙점 계산
    middle_x = new_crosspts[0][0] + (new_crosspts[-1][0] - new_crosspts[0][0]) // 2
    new_crosspts.sort(key=lambda x: x[1])
    middle_y = new_crosspts[0][1] + (new_crosspts[-1][1] - new_crosspts[0][1]) // 2

    # 중앙점을 기준으로 각 사분면 방향으로 멀리 떨어져 있는 좌표 찾기
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    for xy in new_crosspts:
        if xy[0] >= middle_x and xy[1] >= middle_y:
            list_1.append(xy)
        elif xy[0] <= middle_x and xy[1] >= middle_y:
            list_2.append(xy)
        elif xy[0] <= middle_x and xy[1] <= middle_y:
            list_3.append(xy)
        elif xy[0] >= middle_x and xy[1] <= middle_y:
            list_4.append(xy)
        else:
            print("그 게 말 이 돼 ?")

    list_1.sort(key=lambda x: abs(x[0] - middle_x) + abs(x[1] - middle_y))
    list_2.sort(key=lambda x: abs(x[0] - middle_x) + abs(x[1] - middle_y))
    list_3.sort(key=lambda x: abs(x[0] - middle_x) + abs(x[1] - middle_y))
    list_4.sort(key=lambda x: abs(x[0] - middle_x) + abs(x[1] - middle_y))

    final_points = [list_1[-1],
                    list_2[-1],
                    list_3[-1],
                    list_4[-1]]

    for xy in final_points:
        cv2.circle(img, (xy[0], xy[1]), 5, (255, 0, 0), 3)
    # cv2.imshow('img1', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(filepath + "../test/test_4point.jpg", img)

    # 드디어 4개의 점을 받았다 원근변환을 시행하자
    # pts1 = np.float32([])
    pts2 = np.float32(
        [(0, 0), (x_size, 0), (0, y_size), (x_size, y_size)])

    final_points.sort(key=lambda x: (x[1], x[0]))

    test1 = final_points[:2]
    test1.sort(key=lambda x: x[0])
    test2 = final_points[2:]
    test2.sort(key=lambda x: x[0])

    final_points = test1 + test2
    pts1 = np.float32(final_points)

    # 원근 투시(투영) 변환 실행
    perspect_mat = cv2.getPerspectiveTransform(pts1, pts2)  # .astype('float32')
    dst1 = cv2.warpPerspective(img_cp, perspect_mat, (x_size, y_size), cv2.INTER_CUBIC)

    # cv2.imshow("dst1", dst1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(filepath + "../test/test_4point_2.jpg", dst1)
    return dst1.copy()


def find_WeldingLine(img_workspace):
    # TODO: 조동휘 JOB
    model = YOLO('./best.pt')
    result = model.predict(source=img_workspace, save=False, show=False, stream=True, save_txt=False, device='cpu')

    # Kmeans
    src = img_workspace.copy()

    data = src.reshape((-1, 3)).astype(np.float32)

    K = 5
    term_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(data, K, None, term_crit, 5,
                                      cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape(src.shape)

    # find line
    for r in result:
        find = np.zeros((img_workspace.shape[0], img_workspace.shape[1]), dtype=np.uint8)
        mask = np.zeros((img_workspace.shape[0], img_workspace.shape[1]), dtype=np.uint8)
        # r = r.cpu()
        # r = r.numpy()

        box = r.boxes.xyxy[0].cpu().numpy()
        pos = r.masks.xy[0]

        print(box)

        mask = cv2.fillPoly(mask, [pos.astype(np.int32)], (255))
        find = cv2.polylines(find, [pos.astype(np.int32)], True, (255))

        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        canny = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 21)
        # canny = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
        # canny = cv2.Canny(gray, 200, 250)

        img_workspace = cv2.polylines(img_workspace, [pos.astype(np.int32)], True, (255, 0, 0))

        height = int((box[3] - box[1]) / 15)
        width = height

        windows_pos_x = box[0]
        windows_pos_y = box[1]

        plot_x = []
        plot_y = []

        while windows_pos_y + height < box[3]:
            while windows_pos_x + width < box[2]:
                current_window = canny[int(windows_pos_y):int(windows_pos_y + height),
                                 int(windows_pos_x):int(windows_pos_x + width)]
                hist = cv2.calcHist([current_window], [0], None, [256], [0, 255])

                plot_x.append(windows_pos_x)
                sum = hist[220:256].sum()
                plot_y.append(sum)

                # print(f'current pos : w={windows_pos_x}, h={windows_pos_y}')
                # print(f'sum = {sum}')

                # cv2.imshow('curr', current_window)
                # cv2.waitKey(0)
                windows_pos_x = windows_pos_x + 1

            # plt.bar(plot_x, plot_y)
            # plt.show()

            # cv2.imshow('current', canny[int(windows_pos_y):int(windows_pos_y + height), int(box[0]):int(box[2])])
            # cv2.waitKey(0)

            plot_x = []
            plot_y = []
            windows_pos_x = box[0]
            windows_pos_y = windows_pos_y + height

        # cv2.imshow("find", find[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
        # cv2.imshow("img", img_workspace[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
        # cv2.imshow("canny", canny[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
        # cv2.imshow("and", cv2.bitwise_or(find[int(box[1]):int(box[3]), int(box[0]):int(box[2])],
        #                                  canny[int(box[1]):int(box[3]), int(box[0]):int(box[2])]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return mask.copy()

class WorkspaceFinder:
    def __init__(self, img, size_x=1280, size_y=720):
        self.img = img
        self.size_x = size_x
        self.size_y = size_y

        workspace, err, err_pt = self.tryFindWorkspace()
        weldingline_mask = None
        if err == 200:
            weldingline_mask, err, err_pt = self.requestMask(workspace.copy())

        self.send_data = {
            'img_workspace': workspace,
            'weldingline_mask': weldingline_mask,
            'error_messege': err,
            'error_point': err_pt
        }

    # tryFindWorkspace
    def tryFindWorkspace(self):
        # first: 시편 영역 검출
        err = 200
        err_pt = ""
        try:
            dst1 = find_Workspace(self.img, self.size_x, self.size_y)

        except Exception as ex:  # 에러 종류
            print('에러가 발생 했습니다', ex)  # ex는 발생한 에러의 이름을 받아오는 변수
            dst1 = np.zeros((self.size_y, self.size_x))
            err = 404
            err_pt = "tryFindWorkspace(self)"
        return dst1, err, err_pt

    # requestMask
    def requestMask(self, img_workspace):
        err = 200
        err_pt = ""
        try:
            weldingline_mask = find_WeldingLine(img_workspace)
        except Exception as ex:  # 에러 종류
            print('에러가 발생 했습니다', ex)  # ex는 발생한 에러의 이름을 받아오는 변수
            weldingline_mask = np.zeros((self.size_y, self.size_x))
            err = 404
            err_pt = "requestMask(self, img_workspace)"
        return weldingline_mask, err, err_pt

    # applyMask
    def thirdJob(self):
        pass

    def getData(self):
        return self.send_data

