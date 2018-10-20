from six.moves import xrange
import numpy as np
import tensorflow as tf
import random
import os
from PIL import Image
from tqdm import tqdm

class Omniglot():
    def __init__(self,
                 data_dir='datasets/omniglot',
                 image_size=(28,28),
                 seed=0):

        def _full_path(dir):
            return [ os.path.join(dir,f) for f in os.listdir(dir) ]

        alphabets = [ (os.path.basename(f),f) for f in _full_path(os.path.join(data_dir,'images_background')) if os.path.isdir(f) ] + \
                    [ (os.path.basename(f),f) for f in _full_path(os.path.join(data_dir,'images_evaluation')) if os.path.isdir(f) ]
        chars = [
            (alphabet+'_' +os.path.basename(f),f)
            for alphabet,path in alphabets
                for f in _full_path(path)
                    if os.path.isdir(f)
        ]

        old_state = random.getstate()
        random.seed(seed)
        random.shuffle(chars)
        random.setstate(old_state)

        chars = [ (name,[f for f in _full_path(path) if os.path.splitext(f)[1] == '.png']) for name,path in chars]

        assert( len(alphabets) == 50)
        assert( len(chars) == 1623)
        for name,filenames in chars :
            assert(len(filenames) == 20)

        self.chars = chars
        self.train_chars = chars[:1200]
        self.valid_chars = chars[1200:]

    def build_queue(self,task_num,n_way,k_shots,train=True,num_threads=1):
        chars = self.train_chars if train else self.valid_chars

        with tf.device('/cpu'):
            # Load all images to memory for faster reading.
            chars_cache_idx = []
            ims = np.zeros((len(chars)*20,28,28,1),np.float32)
            cnt = 0
            for (name,files) in tqdm(chars):
                idxes = []
                for fnames in files :
                    resized = Image.open(fnames).resize((28,28),resample=Image.LANCZOS)
                    ims[cnt] = np.expand_dims(np.asarray(resized,np.float32),3)
                    idxes.append(cnt)
                    cnt += 1
                chars_cache_idx.append((name,idxes))
            ims = tf.convert_to_tensor(ims)

            def _get_single_task(n_way,k_shots):
                idxes = np.random.choice( len(chars), n_way, replace=False)
                rots = np.random.choice( 4, n_way, replace=True )

                names = [ chars_cache_idx[idx][0] for idx in idxes]
                files = [ np.random.choice(chars_cache_idx[idx][1],k_shots*2,replace=False)
                        for idx in idxes ]
                files = np.stack(files,axis=0)
                rots = np.tile( rots.reshape(n_way,1),[1,k_shots] )
                labels = np.tile( np.arange(0,n_way).reshape(n_way,1),[1,k_shots*2] )

                x,x_prime = np.split(files, 2, axis=1)
                y,y_prime = np.split(labels,2, axis=1)

                rots = rots.reshape(-1)
                x = x.reshape(-1)
                x_prime = x_prime.reshape(-1)
                y = y.reshape(-1)
                y_prime = y_prime.reshape(-1)
                return np.array(names,np.string_),rots,x,y,x_prime,y_prime

            def _read_single_im(elems):
                f_idx,rot = elems #file index, rotations
                _t = ims[f_idx]

                def _raise():
                    assert_op = tf.Assert(False,['Undefined Rotation'])
                    with tf.control_dependencies([assert_op]):
                        return _t
                _t = tf.case({
                    tf.equal(rot,0): lambda : _t,
                    tf.equal(rot,1): lambda : tf.image.rot90(_t,1),
                    tf.equal(rot,2): lambda : tf.image.rot90(_t,2),
                    tf.equal(rot,3): lambda : tf.image.rot90(_t,3)},
                    default= _raise)

                #_t = tf.cast(_t,tf.float32) / 255.0 # Omniglot is already [0-1] since images are BW
                #_t = tf.subtract(_t, 0.5)
                #_t = tf.multiply(_t, 2.0)
                return tf.transpose(_t,[2,0,1])

            task,rots,x,y,x_prime,y_prime = tf.py_func(_get_single_task,
                                                        [n_way,k_shots],
                                                        [tf.string,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64],
                                                        stateful=True)
            task = tf.reshape(task,(n_way,))
            x = tf.reshape(x,(n_way*k_shots,))
            y = tf.reshape(y,(n_way*k_shots,))
            x_prime = tf.reshape(x_prime,(n_way*k_shots,))
            y_prime = tf.reshape(y_prime,(n_way*k_shots,))

            x = tf.map_fn(_read_single_im,[x,rots],dtype=tf.float32,back_prop=False,parallel_iterations=1)
            x_prime = tf.map_fn(_read_single_im,[x_prime,rots],dtype=tf.float32,back_prop=False,parallel_iterations=1)

            # Build task batch
            tasks, x, y, x_prime, y_prime = tf.train.batch(
                [task,x,y,x_prime,y_prime],
                batch_size=task_num,
                num_threads=num_threads,
                capacity=10*task_num)

            return tasks,x,y,x_prime,y_prime