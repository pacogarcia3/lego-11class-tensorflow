import label_image_fg


BASE_PATH='F:/tensorflow-train-1'

TF_PATH_0=BASE_PATH+'/tf_files_originID_4cam_Plus180'
TF_PATH_1=BASE_PATH+'/tf_files_originID_4cam_Plus'
TF_PATH_2=BASE_PATH+'/tf_files_originID_4camONLY'

IMAGE_PATH_A=BASE_PATH+'/Eval_0'
IMAGE_PATH_B=BASE_PATH+'/Eval_r0'
IMAGE_PATH_C=BASE_PATH+'/Eval_All'

CSV_PATH=BASE_PATH

label_image_fg.batch_label_image(TF_PATH_2,IMAGE_PATH_A,CSV_PATH)
