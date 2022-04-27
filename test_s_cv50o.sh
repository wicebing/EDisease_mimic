echo "====== test SPEC ? ?  ======" $1 $2 

python EDisease_train_mimic_s_cv50.py test_ $1 0 origin

python EDisease_train_mimic_s_cv50.py test_ $1 1 origin

python EDisease_train_mimic_s_cv50.py test_ $1 2 origin

python EDisease_train_mimic_s_cv50.py test_ $1 0 Less50

python EDisease_train_mimic_s_cv50.py test_ $1 1 Less50

python EDisease_train_mimic_s_cv50.py test_ $1 2 Less50
