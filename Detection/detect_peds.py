import cv2

def main():

    cap = cv2.VideoCapture('test_videos/ped_test.mp4')
    ped_cascade = cv2.CascadeClassifier('model/ped.xml')

    while True:
        ret, frames = cap.read()
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        cars = ped_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in cars:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow('Peds', frames)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
