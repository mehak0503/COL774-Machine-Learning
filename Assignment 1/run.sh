if [ $# == 4 ]
then
	python q$1.py $2 $3 $4
elif [ $# == 5 ]
then
	python q$1.py $2 $3 $4 $5
elif [ $# == 3 ]
then
	python q$1.py $2 $3
else
	echo "Incorrect number of arguments"
fi	

