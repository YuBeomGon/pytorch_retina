# 사용 안할 annotation 목록
rej_table = [
    '삭제', 'pulled pork', 'abnormal', 'dog', 
    'ASCUS-RE',
    'ASCUS-Re',
    'ASCUS-SIL',
    'ASCwUS-SIL',
    'ASCUS-H',
    'ASC-H',
 
    'AGUS',
    'Reactive cell',
    'Reactive change',    

    'cavity',
    'ASCUS-koilocyte',
    'Lymphocytes',
    'leukocyte',
    'Lymphocyte',
    'leukocytes',
]

# 기 작성된 annotation에서 학습하고자 하는 단위로 교체
replace_table = { 
    'Normal-endocervical cells': 'Normal',
    'Normal-Autolytic parabasal cells': 'Normal',
    'Metaplastic cell-Nomal': 'Normal',
    'Normal-multi-nuclear cell': 'Normal',
    'Normal-metaplastic cell': 'Normal',
    'Normal-parabasal cell': 'Normal',
    'Normal-parabasal cells': 'Normal',
    'Normal-parabasal cells ': 'Normal',
    'Normal-Endocervical cell': 'Normal',
    'Endocervical cell-Normal': 'Normal',
    'Endocervical cell': 'Normal',
    'Endometrial cell': 'Normal',
    'Metaplastic cell': 'Normal',
    'Parabasal cell': 'Normal',
    'No malignant cell': 'Normal',
    'No nalignant cell': 'Normal',
    'No malinant cell': 'Normal',
    'No malignant cell-tissue repair': 'Normal',
    'No malignant cell-endocervical cell': 'Normal',
    'No malinant cell-endocervical cell': 'Normal',
    'No malignant cell-squamous metaplasia': 'Normal',
    'No maligant cell-parabasal cell': 'Normal',
    'No maligant cell-squamous metaplasia cell': 'Normal',
    'No maligant cell-endocervical cell': 'Normal',
    'No malignant cell-Squamous metaplastic cell': 'Normal',
    'No maligant cell-squamous metaplastic cell': 'Normal',
    'No malignant cell-metaplastic cell': 'Normal',

    'No malignant cell-Parabasal cell': 'Normal',
    'No malignant cell-parabasal cell': 'Normal',
    'No malinant cell-parabasal cell': 'Normal',
    'Autolytic parabasal cell': 'Normal',

    'normal': 'Normal',
    'ASCUS-US': 'ASCUS',
    
    'HSILw': 'HSIL',
    
    'Adenocarcinoma': 'Carcinoma',
    'Adenocarcinoma-endocervical type': 'Carcinoma',
    'Adenocarcinoma-endometrial type': 'Carcinoma',

    'Squamous metaplastic cell': 'Normal',
    
    'Squamous cell carcinomaw': 'Carcinoma',
    'Squamous cell carcinoma': 'Carcinoma',
    'Suamous cell carcinoma': 'Carcinoma',
    'Squamous cell carcinama': 'Carcinoma',
    'Normal' : 'Normal',
    'Abnormal' : 'Abnormal',
    'HSIL' : 'HSIL',
    'LSIL' : 'LSIL',
    
    'ASCUS' : 'ASCUS',
    'Carcinoma' : 'Carcinoma',
}

# 사용 안할 이미지 크기
rej_size = [
    (3024, 4032)
]