# Define the Python interpreter
PYTHON = python3
MPIRUN = mpirun 

# Default target
all: main main1 main2 main3 main4 main5 main6 main7 main8 main9 main10 main11 main12 main13

# Target for running main.py
main:
	$(PYTHON) src/main.py

# Target for running main1.py
main1:
	$(PYTHON) src/main1.py

# Target for running main2.py
main2:
	$(PYTHON) src/main2.py

# Target for running main3.py
main3:
	$(PYTHON) src/main3.py

# Target for running main4.py
main4:
	$(PYTHON) src/main4.py

# Target for running main5.py
main5:
	$(PYTHON) src/main5.py

# Target for running main6.py
main6:
	$(PYTHON) src/main6.py

# Target for running main7.py
main7:
	$(PYTHON) src/main7.py

# Target for running main8.py
main8:
	$(PYTHON) src/main8.py

# Target for running main9.py
main9:
	$(PYTHON) src/main9.py

# Target for running main10.py
main10:
	$(PYTHON) src/main10.py

# Target for running main11.py
main11:
	$(PYTHON) src/main11.py

# Target for running main11.py
main12:
	$(PYTHON) src/main12.py

# Target for running main11.py
main13:
	$(PYTHON) src/main13.py
