# 驗證碼生成器，批量生成本地文件，用於訓練
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import os 
import sys
import random

image = ImageCaptcha()

# 驗證碼中的字符，本程式先只用數字產生驗證碼
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#            'v', 'w', 'x', 'y', 'z']
#ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
#            'V', 'W', 'X', 'Y', 'Z']

# 檢查是否有captcha這個資料夾，若沒有則創造一個
try:
    os.makedirs('./captcha/images')
except FileExistsError:
    None
    
captcha_size = 4 # 驗證碼一般都無視大小寫；驗證碼長度4個字符
num = 3000 # 批量生成num個image
for i in range(0 , num):    
    # 獲取隨機生成的驗證碼
    captcha_text = [random.choice(number) for i in range(captcha_size)]
    
    # 把驗證碼列表轉為字符串
    captcha_text = ''.join(captcha_text)
    
    while captcha_text + '.jpg' in os.listdir('captcha/images'):
        captcha_text = [random.choice(number) for i in range(captcha_size)]
        captcha_text = ''.join(captcha_text)
    
    # 生成驗證碼
    image.write(captcha_text , 'captcha/images/' + captcha_text + '.jpg')
    
    sys.stdout.write('\r>> Converting image : {}/{}'.format(i + 1 , num ))
    sys.stdout.flush()
    
print('生成完畢')