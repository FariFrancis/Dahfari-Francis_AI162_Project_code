import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime


class FaceRecognizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # FPS variables
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # Frame count
        self.frame_count = 0

        # Known faces database
        self.known_face_features = []
        self.known_face_names = []

        # Face tracking variables
        self.last_frame_centroids = []
        self.current_frame_centroids = []
        self.last_frame_names = []
        self.current_frame_names = []

        # Database connection
        self.conn = sqlite3.connect("attendance.db")
        self.cursor = self.conn.cursor()

        # Create attendance table
        current_date = datetime.datetime.now().strftime("%Y_%m_%d")
        self.table_name = "attendance_" + current_date
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {self.table_name} (name TEXT, time TEXT, date DATE, UNIQUE(name, date))"
        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def load_known_faces(self):
        if os.path.exists("data/features_all.csv"):
            features_csv = pd.read_csv("data/features_all.csv", header=None)
            for i in range(features_csv.shape[0]):
                features = []
                self.known_face_names.append(features_csv.iloc[i][0])
                for j in range(1, 129):
                    if features_csv.iloc[i][j] == '':
                        features.append('0')
                    else:
                        features.append(features_csv.iloc[i][j])
                self.known_face_features.append(features)
            logging.info("Loaded {} faces from database.".format(len(self.known_face_features)))
            return True
        else:
            logging.warning("Features file 'features_all.csv' not found!")
            return False

    def update_fps(self):
        now = time.time()
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def calculate_distance(self, feature1, feature2):
        feature1 = np.array(feature1)
        feature2 = np.array(feature2)
        distance = np.sqrt(np.sum(np.square(feature1 - feature2)))
        return distance

    def track_faces(self, faces):
        pass

    def recognize_faces(self, faces):
        pass

    def mark_attendance(self, name):
        pass

    def draw_info(self, frame):
        pass

    def process_video(self, stream):
        if self.load_known_faces():
            while stream.isOpened():
                self.frame_count += 1
                logging.debug("Processing frame {}".format(self.frame_count))
                flag, frame = stream.read()
                key = cv2.waitKey(1)

                faces = []  # Placeholder for face detection results

                self.last_frame_centroids = self.current_frame_centroids
                self.current_frame_centroids = []

                self.last_frame_names = self.current_frame_names[:]
                self.current_frame_names = []

                # Main processing logic
                # ...

                if key == ord('q'):
                    break

                self.update_fps()
                cv2.imshow("camera", frame)

                logging.debug("Frame processing complete\n\n")

        else:
            logging.error("Failed to load known faces. Exiting...")
            return

    def run(self):
        cap = cv2.VideoCapture(0)  # Get video stream from camera
        self.process_video(cap)
        cap.release()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO)
    face_recognizer = FaceRecognizer()
    face_recognizer.run()


if __name__ == '__main__':
    main()