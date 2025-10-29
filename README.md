3. TensorFlow Datasets 기본 사용
import tensorflow_datasets as tfds



# 패션 MNIST 데이터셋 불러오기
data = tfds.load('fashion_mnist', with_info=True, as_supervised=True)
train_data, test_data = data['train'], data['test']

print(train_data)



🔍 실행 결과

train_data와 test_data로 자동 분리
각 데이터는 (image, label) 쌍으로 구성된 tf.data.Dataset 객체
MNIST의 경우: 28×28 흑백 이미지 60,000장 + 테스트 10,000장




4. 케라스 모델에 데이터셋 연결
import tensorflow as tf
import tensorflow_datasets as tfds

(train_images, train_labels), (test_images, test_labels) = \
  tfds.as_numpy(tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True, batch_size=-1))

train_images = tf.cast(train_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)





📈 결과 요약

정확도 약 0.88~0.91
Dropout(0.2)로 과적합 완화
TFDS 데이터셋은 케라스 모델에 바로 연결 가능



5. 데이터 증식(Augmentation) 적용하기
def augmentimages(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_flip_left_right(image)
    return image, label

train_data = train_data.map(augmentimages)





⚙️ 설명

map() 함수를 통해 데이터셋 전체에 증강 함수를 적용

tf.image.random_flip_left_right() : 좌우 반전

데이터 다양성을 높여 과적합 방지 효과

6. TensorFlow Addons를 활용한 고급 증강
import tensorflow_addons as tfa

def augmentimages(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')
    return image, label




🔍 결과

회전 + 반전 등 복합 증강으로 데이터 다양성 강화

모델이 더 일반화되어 테스트 정확도 2~4% 상승




7. 사용자 정의 데이터 분할
data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
val_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)
test_data = tfds.load('cats_vs_dogs', split='train[90%:]', as_supervised=True)




📘 설명

split 구문으로 데이터셋 일부만 로드 가능

비율(:80%, 80%:90%) 또는 개수(:10000, -1000:) 지정

모델 학습/검증/테스트 세트를 유연하게 구성 가능

8. TFRecord 이해하기
data, info = tfds.load('mnist', with_info=True)
print(info)

filename = '/root/tensorflow_datasets/mnist/3.0.1/mnist-train.tfrecord-00000-of-00001'
raw_dataset = tf.data.TFRecordDataset(filename)

for raw_record in raw_dataset.take(1):
    print(repr(raw_record))




🔎 개념 요약

TFRecord: TensorFlow 전용 바이너리 데이터 포맷

대규모 데이터 저장 시 메모리 효율적

tf.data.TFRecordDataset()으로 직접 읽기 가능




💡 실습 결과

출력: 이미지 바이너리(PNG) + 라벨 정보

빠른 로딩 




9. ETL 프로세스 (Extract, Transform, Load)
🧾 개념 요약

ETL은 데이터 처리 파이프라인의 핵심

단계	설명
추출(Extract)	원본 데이터를 로드 (TFDS, 로컬 파일 등)
변환(Transform)	증강, 정규화, 전처리 수행
로드(Load)	GPU/TPU에 배치 전달 및 학습 수행
10. ETL 전체 예제 요약
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

data = tfds.load('horses_or_humans', split='train', as_supervised=True)
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)

def augment(image, label):
    image = tf.cast(image, tf.float32)/255.0
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')
    return image, label

train = data.map(augment).shuffle(100).batch(32)
val = val_data.batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, epochs=10, validation_data=val)

📈 실행 결과

증강 적용 시 일반화 성능 향상



11. 로드 단계 최적화 (CPU → GPU/TPU 파이프라인)

데이터 추출/변환은 CPU,
로드 및 학습은 GPU/TPU에서 수행

tf.data의 파이프라인 구조를 통해 I/O 병목을 최소화

prefetch()와 cache()를 활용하면 학습 속도를 향상

12. 💡 실습 요약 및 시사점
항목	내용
데이터 자동화	tfds.load()로 데이터 로드 및 분할 자동화
증강 처리	tf.image, tfa.image로 간단한 이미지 증식
효율적 저장	TFRecord를 통한 대용량 데이터 관리
성능 최적화	ETL 기반 파이프라인으로 병렬 처리
학습 효율	데이터 준비 과정의 단순화 및 모델 일반화 향상


✅ 결론

TensorFlow Datasets를 사용하면
1️⃣ 데이터 준비 시간 단축,
2️⃣ 전처리 및 증강 자동화,
3️⃣ 모델 일반화 성능 향상,
4️⃣ 대규모 데이터 관리 효율 극대화

이 장의 핵심 : 데이터셋을 손쉽게 불러오고, 자동화된 파이프라인으로 학습 효율을 높이는 것
