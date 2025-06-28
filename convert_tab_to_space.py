# 처리할 원본 파일 경로
input_file_path = '/aivanas/raw/action/violence/action_recognition/data/UCF_Crime/annotations_orign.txt'
# 결과를 저장할 파일 경로
output_file_path = '/aivanas/raw/action/violence/action_recognition/data/UCF_Crime/annotations.txt'

try:
    # 원본 파일을 읽기 모드로, 새 파일을 쓰기 모드로 연다
    with open(input_file_path, 'r', encoding='utf-8') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
        # 파일의 모든 내용을 한 번에 읽어온다
        content = f_in.read()
        # 읽어온 내용에서 탭(\t)을 공백(' ')으로 모두 변경한다
        modified_content = content.replace('\t', ' ')
        # 변경된 내용을 새 파일에 쓴다
        f_out.write(modified_content)

    print(f"파일 처리 완료: '{input_file_path}'의 내용이 '{output_file_path}'에 저장되었습니다.")

except FileNotFoundError:
    print(f"오류: '{input_file_path}' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류 발생: {e}")