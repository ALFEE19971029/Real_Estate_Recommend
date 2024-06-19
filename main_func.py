import gradio as gr

##파일 경로##
#시각화
import visualization
#각자 파일
import shinwoong.shinwoong_function as ksw
import geunsoo.geunsoo_function as kgs
import hwan.hwan_function as ch
import jiyeon.jiyeon_function as cgy

def process_input(text1, text2, choice):
    lower_limit = int(text1)
    upper_limit = int(text2)
    if choice == "김근수":
        return visualization.visualization(kgs.geunsoo_func(lower_limit, upper_limit))
    elif choice == "김신웅":
        return visualization.visualization(ksw.shinwoong_func(lower_limit, upper_limit))
    elif choice == "최지연":
        return visualization.visualization(cgy.jiyeon_model(lower_limit, upper_limit))
    elif choice == "최환":
        return visualization.visualization(ch.whan_model(lower_limit, upper_limit))

radio_choices = ["김근수", "김신웅", "최지연", "최환"]

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            textbox1 = gr.Textbox(label="최소 가격(만원)")
            textbox2 = gr.Textbox(label="최대 가격(만원)")
            radio = gr.Radio(label="적용 모델", choices=radio_choices)
            submit_button = gr.Button("Submit")
        with gr.Column():
            output = gr.Plot(label="Output")

    submit_button.click(process_input, inputs=[textbox1, textbox2, radio], outputs=output)

demo.launch()
