// 마이크 입력을 가공 없이 메인 스레드로 넘긴다
class Recorder extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice());
    return true;
  }
}
registerProcessor("rec", Recorder);
