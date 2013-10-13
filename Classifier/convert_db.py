__author__ = 'Mario'

import sqlite3
import cv2 as cv

if __name__ == '__main__':

    read_path = '../Databases/LRS/aflw/data/flickr'
    write_path = '../Databases/LRS/faces'
    db_path = '../Databases/LRS/aflw/data/aflw.sqlite'

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    query = 'SELECT f.face_id, fr.x, fr.y, fr.w, fr.h, fi.filepath  FROM FaceImages as fi, Faces as f, FaceRect as fr  WHERE fi.file_id = f.file_id AND fr.face_id = f.face_id'

    i = 0
    for row in c.execute(query):

        id = row[0]
        x = max(row[1], 0)
        y = max(row[2], 0)
        w = row[3]
        h = row[4]
        f = row[5]

        im = cv.imread('%s/%s' % (read_path, f))
        dh = min(y+h, im.shape[0]-1)
        dw = min(x+w, im.shape[1]-1)

        w = dw - x
        h = dh - y

        if h != w:
            if w > h:
                d = w - h
                w = h
                x += d / 2
            else:
                d = h - w
                h = w
                y += d / 2
            dw = x + w
            dh = y + h

        im2 = im[y:dh, x:dw, :]
        im2 = cv.resize(im2, (64, 64))

        cv.imwrite('%s/%s.jpg' % (write_path, id), im2)

        if i % 10 == 0:
            print i, id, f, x, y, w, h, dw, dh, im.shape
        i+=1