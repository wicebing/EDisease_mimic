echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_spectrums.py train_ $1 $2 origin cossin

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_spectrums.py test_ $1 $2 origin cossin

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_s_spectrums.py train_ $1 $2 origin sigmoid

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_s_spectrums.py test_ $1 $2 origin sigmoid

