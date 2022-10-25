# Script to run perceplearn.py and percepclassify.py

if [ "$1" ]; then
  echo "Training on input file: $1.....";
  python3 perceplearn.py $1;
  echo -e "DONE\n"
else
  echo "Training on default input file.....";
  python3 perceplearn.py;
  echo -e "DONE\n"
fi

if [ "$2" ] && [ "$3" ]; then
  echo "Classifying raw input file: $3, using model file: $2";
  python3 percepclassify.py $2 $3;
  echo "Output classification results to percepoutput.txt";
elif [ "$2" ]; then
  echo "Classifying raw input file: dev-text.txt, using model file: $2";
  python3 percepclassify.py $2 dev-text.txt;
  echo "Output classification results to percepoutput.txt";
elif [ "$3" ]; then
  echo "Classifying raw input file: $3, using model file: vanillamodel.txt";
  python3 percepclassify.py vanillamodel.txt $3;
  echo "Output classification results to percepoutput.txt";
else
  echo "Classifying raw input file: dev-text.txt, using model file: vanillamodel.txt";
  python3 percepclassify.py;
  echo "Output classification results to percepoutput.txt";
fi
