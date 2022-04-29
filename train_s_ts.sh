echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic_timesequence.py train_ $1 $2 origin

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic_timesequence.py test_ $1 $2 origin





