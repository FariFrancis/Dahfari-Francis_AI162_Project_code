import os
import dlib
import csv
import numpy as np
import logging
import cv2

class FeatureExtractor:
    def __init__(self):
        self.path_images = "data/faces_from_camera/"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
        self.face_model = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")

    def extract_features(self, img_path):
        img = cv2.imread(img_path)
        faces = self.detector(img, 1)
        logging.info("Faces detected in image: {}".format(img_path))

        if len(faces) != 0:
            shape = self.predictor(img, faces[0])
            features = self.face_model.compute_face_descriptor(img, shape)
        else:
            features = np.zeros(128)
            logging.warning("No face detected in image: {}".format(img_path))

        return features

    def calculate_mean_features(self, person_path):
        features_list = []
        photos_list = os.listdir(person_path)
        for photo in photos_list:
            logging.info("Reading image: {}".format(os.path.join(person_path, photo)))
            features = self.extract_features(os.path.join(person_path, photo))
            if not np.all(features == 0):
                features_list.append(features)

        if features_list:
            mean_features = np.mean(features_list, axis=0)
        else:
            mean_features = np.zeros(128)
            logging.warning("No valid images found in: {}".format(person_path))

        return mean_features

    def extract_and_save_features(self):
        logging.basicConfig(level=logging.INFO)
        person_list = os.listdir(self.path_images)
        person_list.sort()

        with open("data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for person in person_list:
                logging.info("Processing person: {}".format(person))
                mean_features = self.calculate_mean_features(os.path.join(self.path_images, person))

                if len(person.split('_', 2)) == 2:
                    person_name = person
                else:
                    person_name = person.split('_', 2)[-1]

                mean_features = np.insert(mean_features, 0, person_name)
                writer.writerow(mean_features)
                logging.info('\n')
            logging.info("All features saved to: data/features_all.csv")


def main():
    feature_extractor = FeatureExtractor()
    feature_extractor.extract_and_save_features()


if __name__ == '__main__':
    main()