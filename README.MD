Copyright (c) 2017 Hanze Dong. All Rights Reserved.

There are 3 parts in our model:



1. main function:

   ​	main.py and extraction.py (backup)

   ​	mainly deal with the top struct: 

   ​		such as interpolation, parameter adjustment etc.

2. HMM:

   ​	HMM.py

   ​	include the complenent of Hidden Markov Model

   ​	such as building transition matrix, optimization

3. Data Processing:

   ​	read_data.py, decode_pos.py, add_pos.py

   ​	Read data from txt and supplement POS information