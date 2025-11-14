import pexpect

adb_cmd = (
    'adb shell -t -t "export LD_LIBRARY_PATH=/data/local/tmp/MAMBA/:$LD_LIBRARY_PATH; '
    '/data/local/tmp/MAMBA/MambaTest -output-buffer-size 1"'
)

child = pexpect.spawn(adb_cmd, encoding='utf-8', timeout=30)

child.expect(r'\[Input\]:')  # [Input]: 프롬프트 기다림
print(">> Mamba ready!")

child.sendline("Black Hole?")  # 안전하게 입력 전송

while True:
    idx = child.expect([r'\[Input\]:', pexpect.EOF, pexpect.TIMEOUT])
    if idx == 0:
        print(child.before, end='')  # 이전 출력
        child.sendline("Next question")
    elif idx == 1:
        print("Process ended")
        break
    elif idx == 2:
        print("Timeout waiting for input")
