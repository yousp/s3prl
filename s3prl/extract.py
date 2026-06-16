import pandas as pd
import tensorflow as tf

def extract(event_path, data_path):
    data = []
    for event in tf.compat.v1.train.summary_iterator(event_path):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                data.append({
                    'Wall_Time': event.wall_time,
                    'Step': event.step,
                    'Tag': value.tag,
                    'Value': value.simple_value
                })
    df = pd.DataFrame(data)
    print(df.head())
    df.to_csv(data_path, index=False)

extract(input("이벤트 파일 이름: "), input("출력 파일 이름: "))
