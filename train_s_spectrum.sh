echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_spectrums.py train_ $1 $2 origin linear

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_spectrums.py test_ $1 $2 origin linear

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_spectrums.py train_ $1 $2 origin sinh

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_spectrums.py test_ $1 $2 origin sinh

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_spectrums.py train_ $1 $2 origin exp

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_spectrums.py test_ $1 $2 origin exp

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_spectrums.py train_ $1 $2 origin parabolic

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_spectrums.py test_ $1 $2 origin parabolic

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_spectrums.py train_ $1 $2 origin sigmoid

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_spectrums.py test_ $1 $2 origin sigmoid

