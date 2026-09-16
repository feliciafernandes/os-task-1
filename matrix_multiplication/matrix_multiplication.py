import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

GRID_SIZE = 100
MP4_OUTPUT = "matrix_multiplication_100x100.mp4"
GIF_OUTPUT = "matrix_multiplication.gif"

FONT_BOLD = '/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf'
FONT_REGULAR = '/usr/share/fonts/TTF/JetBrainsMonoNLNerdFont-SemiBold.ttf'

font_header = ImageFont.truetype(FONT_BOLD, 22)
font_caption = ImageFont.truetype(FONT_BOLD, 14)
font_stats = ImageFont.truetype(FONT_REGULAR, 13)
font_operator = ImageFont.truetype(FONT_BOLD, 26)


def create_matrices(size):
    mat_x = [[((r * 2 + c * 3) % 9) + 1 for c in range(size)] for r in range(size)]
    mat_y = [[((r * 5 + c * 7) % 9) + 1 for c in range(size)] for r in range(size)]
    return mat_x, mat_y


def evaluate_cell_dot(row_idx, col_idx, mat_x, mat_y):
    val = 0
    for k in range(GRID_SIZE):
        val += mat_x[row_idx][k] * mat_y[k][col_idx]
    return row_idx, col_idx, val


def execute_parallel_multiplication(mat_x, mat_y):
    print("Running 100x100 matrix multiplication...")
    product_matrix = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    event_log = []

    t_start = time.perf_counter()

    with ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(evaluate_cell_dot, r, c, mat_x, mat_y)
            for r in range(GRID_SIZE) for c in range(GRID_SIZE)
        ]

        for fut in as_completed(futures):
            r, c, val = fut.result()
            product_matrix[r][c] = val
            event_log.append((r, c, val))

    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    print(f"Multiplication finished in {t_elapsed:.2f} ms")
    return product_matrix, event_log, t_elapsed


def render_visualization(mat_x, mat_y, result_mat, event_log):
    print("Generating animation...")

    canvas_width = 1260
    canvas_height = 580
    matrix_box = 230

    top_y = 110
    pos_x1 = 50
    pos_op1 = pos_x1 + matrix_box + 25
    pos_x2 = pos_op1 + 45
    pos_op2 = pos_x2 + matrix_box + 25
    pos_x3 = pos_op2 + 45

    img_x = Image.new("RGB", (GRID_SIZE, GRID_SIZE))
    img_y = Image.new("RGB", (GRID_SIZE, GRID_SIZE))
    pix_x = [((mat_x[r][c] * 20) + 30, 70, 150) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    pix_y = [(140, 50, (mat_y[r][c] * 20) + 40) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    img_x.putdata(pix_x)
    img_y.putdata(pix_y)

    res_canvas_data = [(15, 23, 42)] * (GRID_SIZE * GRID_SIZE)
    max_val = max(max(row) for row in result_mat) or 1

    chunk_size = 50
    total_frames = (len(event_log) + chunk_size - 1) // chunk_size

    ffmpeg_process = subprocess.Popen([
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{canvas_width}x{canvas_height}',
        '-pix_fmt', 'rgb24',
        '-r', '20',
        '-i', '-',
        '-an',
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '22',
        '-preset', 'fast',
        MP4_OUTPUT
    ], stdin=subprocess.PIPE)

    for frame in range(total_frames):
        start = frame * chunk_size
        end = min(start + chunk_size, len(event_log))

        cur_r, cur_c = 0, 0
        for i in range(start, end):
            r, c, val = event_log[i]
            cur_r, cur_c = r, c
            norm = val / max_val
            green = int(70 + norm * 150)
            blue = int(180 - norm * 60)
            red = int(20 + norm * 80)
            res_canvas_data[r * GRID_SIZE + c] = (red, green, blue)

        frame_canvas = Image.new("RGB", (canvas_width, canvas_height), color=(15, 23, 42))
        draw = ImageDraw.Draw(frame_canvas)

        draw.text((canvas_width // 2 - 290, 20), "Multithreaded 100x100 Matrix Multiplication", fill=(248, 250, 252), font=font_header)
        draw.text((canvas_width // 2 - 250, 55), "Every Multiplication Operation Executed as an Independent Task", fill=(148, 163, 184), font=font_stats)

        disp_x = img_x.resize((matrix_box, matrix_box), Image.NEAREST)
        frame_canvas.paste(disp_x, (pos_x1, top_y))
        draw.rectangle((pos_x1 - 2, top_y - 2, pos_x1 + matrix_box + 2, top_y + matrix_box + 2), outline=(51, 65, 85), width=2)
        highlight_y = top_y + int((cur_r / GRID_SIZE) * matrix_box)
        draw.rectangle((pos_x1, highlight_y, pos_x1 + matrix_box, highlight_y + 2), outline=(239, 68, 68), width=2)
        draw.text((pos_x1 + 45, top_y + matrix_box + 12), "Input Matrix A (100x100)", fill=(203, 213, 225), font=font_caption)

        draw.text((pos_op1 + 8, top_y + 90), "x", fill=(56, 189, 248), font=font_operator)

        disp_y = img_y.resize((matrix_box, matrix_box), Image.NEAREST)
        frame_canvas.paste(disp_y, (pos_x2, top_y))
        draw.rectangle((pos_x2 - 2, top_y - 2, pos_x2 + matrix_box + 2, top_y + matrix_box + 2), outline=(51, 65, 85), width=2)
        highlight_x = pos_x2 + int((cur_c / GRID_SIZE) * matrix_box)
        draw.rectangle((highlight_x, top_y, highlight_x + 2, top_y + matrix_box), outline=(239, 68, 68), width=2)
        draw.text((pos_x2 + 45, top_y + matrix_box + 12), "Input Matrix B (100x100)", fill=(203, 213, 225), font=font_caption)

        draw.text((pos_op2 + 8, top_y + 90), "=", fill=(56, 189, 248), font=font_operator)

        img_z = Image.new("RGB", (GRID_SIZE, GRID_SIZE))
        img_z.putdata(res_canvas_data)
        disp_z = img_z.resize((matrix_box, matrix_box), Image.NEAREST)
        frame_canvas.paste(disp_z, (pos_x3, top_y))
        draw.rectangle((pos_x3 - 2, top_y - 2, pos_x3 + matrix_box + 2, top_y + matrix_box + 2), outline=(51, 65, 85), width=2)
        active_px = pos_x3 + int((cur_c / GRID_SIZE) * matrix_box)
        active_py = top_y + int((cur_r / GRID_SIZE) * matrix_box)
        draw.rectangle((active_px - 1, active_py - 1, active_px + 3, active_py + 3), outline=(250, 204, 21), width=2)
        draw.text((pos_x3 + 30, top_y + matrix_box + 12), "Product Matrix C (100x100)", fill=(74, 222, 128), font=font_caption)

        stats_y = 405
        draw.rounded_rectangle((50, stats_y, canvas_width - 50, stats_y + 55), radius=8, fill=(30, 41, 59), outline=(51, 65, 85))
        draw.text((70, stats_y + 18), "Total Multiplication Operations: 10,000", fill=(248, 250, 252), font=font_stats)
        draw.text((460, stats_y + 18), f"Active Operation: Cell C[{cur_r}][{cur_c}]", fill=(251, 191, 36), font=font_stats)
        draw.text((820, stats_y + 18), f"Completed Tasks: {end:,} / 10,000", fill=(74, 222, 128), font=font_stats)

        bar_x = 50
        bar_y = 485
        bar_w = 1160
        progress_pct = end / len(event_log)
        draw.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + 12), fill=(30, 41, 59), outline=(51, 65, 85))
        draw.rectangle((bar_x, bar_y, bar_x + int(bar_w * progress_pct), bar_y + 12), fill=(56, 189, 248))
        draw.text((bar_x, bar_y + 22), f"Progress: {end:,} / 10,000 Operations ({progress_pct*100:.1f}%)", fill=(203, 213, 225), font=font_stats)

        ffmpeg_process.stdin.write(frame_canvas.tobytes())

    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    gif_cmd = [
        'ffmpeg', '-y',
        '-i', MP4_OUTPUT,
        '-vf', 'fps=15,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3',
        GIF_OUTPUT
    ]
    subprocess.run(gif_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print("Files created: " + MP4_OUTPUT + ", " + GIF_OUTPUT)


def main():
    mat_x, mat_y = create_matrices(GRID_SIZE)
    result_mat, event_log, runtime = execute_parallel_multiplication(mat_x, mat_y)
    render_visualization(mat_x, mat_y, result_mat, event_log)


if __name__ == "__main__":
    main()
