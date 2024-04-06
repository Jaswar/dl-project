from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
import numpy as np
import os


def main():
    in_path = 'predictions'
    out_path = 'svm_coefs'

    y = np.load(os.path.join(in_path, 'y.npy'))
    latents = np.load(os.path.join(in_path, 'latents.npy'))

    x_train, x_val, y_train, y_val = train_test_split(latents, y, test_size=0.2, random_state=42)

    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x_train, y_train)
    print('Dummy accuracy:', dummy.score(x_val, y_val))

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(x_train, y_train)
    print('Train accuracy:', svm.score(x_train, y_train))
    print('Validation accuracy:', svm.score(x_val, y_val))

    coef = np.array(svm.coef_).reshape(-1)
    coef = coef / np.linalg.norm(coef)

    np.save(os.path.join(out_path, 'coef.npy'), coef)


if __name__ == '__main__':
    main()

