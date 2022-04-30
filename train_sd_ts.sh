echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic.py trainTS $1 1 origin

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic.py testTS $1 1 origin

echo "====== train SPEC ? ? ======" $1 $2
python EDisease_train_mimic.py trainTS $1 2 origin

echo "====== test SPEC ? ? EM ======" $1 $2
python EDisease_train_mimic.py testTS $1 2 origin





