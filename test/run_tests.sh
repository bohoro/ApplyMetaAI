pause_me(){
    echo "Press any key to continue..."
    read -n 1 -s
}
echo python test_pytorch.py
python test_pytorch.py
pause_me
echo python ../GALACTICA/test_inference.py 
python ../GALACTICA/test_inference.py
pause_me
echo python ../Synthetic/openai_api_poc.py
python ../Synthetic/openai_api_poc.py
pause_me
echo ../Synthetic/openai_detector_poc.py
python ../Synthetic/openai_detector_poc.py
pause_me
echo python ../Synthetic/test_st_inference.py
python ../Synthetic/test_st_inference.py

