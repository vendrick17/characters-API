import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
import PIL
import torch
#from models.stylegan2ada import generate_web, projector_web
from models.stylegan2ada_pytorch.projector import project
from models.stylegan2ada_pytorch import dnnlib
from models.stylegan2ada_pytorch import legacy


import numpy as np
import tqdm
import random


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = './static/uploads'

# Auxiliar functions for the networks

def init_network(network_pkl, seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    return G

def init_projector(target_fname, G):
    device = torch.device('cuda')
    num_steps=100
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
        )
    return  target_pil, target_uint8, projected_w_steps

def validate_image(stream):
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413

@app.route('/')  #the route is called whenever the user sends a GET request
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html') 

@app.route('/', methods=['POST']) #the route is called whenever the user sends POST  request
def upload_files():
    uploaded_names = []
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            return "Invalid image", 400
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    #uploaded_names = os.listdir(app.config['UPLOAD_PATH'])
    #return render_template('form.html', unames=uploaded_names)
    return '', 204

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename) 


@app.route('/silhouette') 
def get_ses_1(): 
 	return render_template('silhouette.html')

@app.route('/character') 
def get_ses_2(): 
 	return render_template('character.html')


@app.route('/generate_random', methods=['GET','POST'])
def generate_random():
    print('generate')
    trained_models_path = './models/stylegan2ada/training-runs/00009-characters_v5-auto1-resumeffhq1024'
    pkl_filename = 'network-snapshot-001520.pkl'
    pkl_path = os.path.join(trained_models_path,pkl_filename)

    _, _, Gs, proj = init_network(pkl_path, seed=322)
    proj.num_steps = 30

    image_names = random.choices(os.listdir('./static/random'), k = 5)
    target_fnames = [os.path.join('./static/random', f'{image_name}') for image_name in image_names]

    out_path = './static/random_generation'
    os.makedirs(out_path, exist_ok=True)

    target_pils = [init_projector(target_fname, Gs)[0] for target_fname in target_fnames]
    target_floats = [init_projector(target_fname, Gs)[1] for target_fname in target_fnames]

    print('projected')
    #seed = 303
    if request.method == 'POST':
        print('request')
        for ix in range(len(image_names)):
            #print(image_names[ix])
            #target_pils[ix].save(f'{out_path}/target_{image_names[ix]}')
            proj.start([target_floats[ix]])

            with tqdm.trange(proj.num_steps) as t:
                for step in t:
                    assert step == proj.cur_step
                    dist, loss = proj.step()
                    t.set_postfix(dist=f'{dist[0]:.4f}', loss=f'{loss:.2f}')

                # Save results.
                PIL.Image.fromarray(proj.images_uint8[0], 'RGB').save(f'{out_path}/proj_{image_names[ix]}')
        projection_names = os.listdir(out_path)
        projection_names = [projection_name for projection_name in projection_names if os.path.splitext(projection_name)[1] in app.config['UPLOAD_EXTENSIONS']]
        
        return render_template('silhouette.html', names=projection_names)


@app.route('/generate_projection', methods=['GET','POST'])
def generate_projection():

    if '1' in request.form.getlist('checkbox_type'):
        trained_models_path_silhouette = './models/stylegan2ada/training-runs/00010-characters_v4_stylegan_2_ch-auto1-ada-target0.7-bgcfnc-resumeffhq1024'
        
        pkl_filename_silhouette = 'network-snapshot-001160.pkl'
        pkl_path_silhouette = os.path.join(trained_models_path_silhouette,pkl_filename_silhouette)
        
        G_silhouette = init_network(pkl_path_silhouette, seed=322)

        ground_truth_path_silhouette = './static/ground_truth/silhouette'
        image_names_silhouette = os.listdir(ground_truth_path_silhouette)
        target_fnames_silhouette = [os.path.join(ground_truth_path_silhouette, f'{image_name}') for image_name in image_names_silhouette]
        target_fnames_silhouette = random.sample(target_fnames_silhouette, 3)
        
        out_path_silhouette = './static/projections/silhouette'
        os.makedirs(out_path_silhouette, exist_ok=True)
        
        target_pils_silhouette, target_uint8_silhouette, projected_w_steps_silhouette = [], [], []
        for target_fname in target_fnames_silhouette:
            pil, uint8, step = init_projector(target_fname, G_silhouette)
            target_pils_silhouette.append(pil)
            target_uint8_silhouette.append(uint8)
            projected_w_steps_silhouette.append(step)

    if '2' in request.form.getlist('checkbox_type'):
        trained_models_path_colored = './models/stylegan2ada/training-runs/00009-characters_v5-auto1-resumeffhq1024'
        
        pkl_filename_colored = 'network-snapshot-001520.pkl'
        pkl_path_colored = os.path.join(trained_models_path_colored,pkl_filename_colored)
        
        G_colored = init_network(pkl_path_colored, seed=322)

        ground_truth_path_colored = './static/ground_truth/colored'
        image_names_colored = os.listdir(ground_truth_path_colored)
        target_fnames_colored = [os.path.join(ground_truth_path_colored, f'{image_name}') for image_name in image_names_colored]
        target_fnames_colored = random.sample(target_fnames_colored, 3)
        
        out_path_colored = './static/projections/colored'
        os.makedirs(out_path_colored, exist_ok=True)
        
        target_pils_colored, target_uint8_colored, projected_w_steps_colored = [], [], []
        for target_fname in target_fnames_colored:
            pil, uint8, step = init_projector(target_fname, G_colored)
            target_pils_colored.append(pil)
            target_uint8_colored.append(uint8)
            projected_w_steps_colored.append(step)
    print(1)
    if request.method == 'POST':
        print(2)
        projection_names_colored, projection_names_silhouette = [], []
        
        if '1' in request.form.getlist('checkbox_type'):
            for ix in range(len(target_fnames_silhouette)):

                target_pils_silhouette[ix].save(f'{out_path_silhouette}/target_{image_names_silhouette[ix]}')
                projected_w = projected_w_steps_silhouette[ix][-1]
                synth_image = G_silhouette.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                PIL.Image.fromarray(synth_image, 'RGB').save(f'{out_path_silhouette}/proj_{ix}.png')
                np.savez(f'{out_path_silhouette}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

            projection_names_silhouette = os.listdir(out_path_silhouette)
            projection_names_silhouette = [projection_name for projection_name in projection_names_silhouette if os.path.splitext(projection_name)[1] in app.config['UPLOAD_EXTENSIONS']]
        if '2' in request.form.getlist('checkbox_type'):

            for ix in range(len(target_fnames_colored)):
                #print(image_names[ix])
                target_pils_colored[ix].save(f'{out_path_colored}/target_{image_names_colored[ix]}')
                
                projected_w = projected_w_steps_colored[ix][-1]
                synth_image = G_silhouette.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                synth_image = (synth_image + 1) * (255/2)
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                PIL.Image.fromarray(synth_image, 'RGB').save(f'{out_path_colored}/proj_{ix}.png')
                np.savez(f'{out_path_colored}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

            projection_names_colored = os.listdir(out_path_colored)
            projection_names_colored = [projection_name for projection_name in projection_names_colored if os.path.splitext(projection_name)[1] in app.config['UPLOAD_EXTENSIONS']]

        return render_template('silhouette.html', names_colored = projection_names_colored, names_silhouette = projection_names_silhouette)

@app.route('/generate_projection_cbw', methods=['GET','POST'])
def generate_projection_cbw():
    if '1' in request.form.getlist('checkbox_type'):
        trained_models_path_silhouette = './models/stylegan2ada/training-runs/00010-characters_v4_stylegan_2_ch-auto1-ada-target0.7-bgcfnc-resumeffhq1024'
        
        pkl_filename_silhouette = 'network-snapshot-001160.pkl'
        pkl_path_silhouette = os.path.join(trained_models_path_silhouette,pkl_filename_silhouette)
        
        _, _, Gs_silhouette, proj_silhouette = init_network(pkl_path_silhouette, seed=322)
        proj_silhouette.num_steps = 50

        ground_truth_path_silhouette = './static/ground_truth/silhouette'
        image_names_silhouette = os.listdir(ground_truth_path_silhouette)
        
        target_fnames_silhouette = [os.path.join(ground_truth_path_silhouette, f'{image_name}') for image_name in image_names_silhouette]
        target_fnames_silhouette = random.sample(target_fnames_silhouette, 3)

        out_path_silhouette = './static/projections/silhouette'
        os.makedirs(out_path_silhouette, exist_ok=True)

        target_pils_silhouette = [init_projector(target_fname, Gs_silhouette)[0] for target_fname in target_fnames_silhouette]
        target_floats_silhouette = [init_projector(target_fname, Gs_silhouette)[1] for target_fname in target_fnames_silhouette]

    if '2' in request.form.getlist('checkbox_type'):

            model_path = './models/pix2pix/training_checkpoints/'
            input_images = os.listdir(app.config['UPLOAD_PATH'])
            input_images_path = [os.path.join(app.config['UPLOAD_PATH'], input_image) for input_image in input_images]
    #seed = 303
    if request.method == 'POST':
        projection_names_silhouette = []

        if '1' in request.form.getlist('checkbox_type'):
            for ix in range(len(target_fnames_silhouette)):

                target_pils_silhouette[ix].save(f'{out_path_silhouette}/target_{image_names_silhouette[ix]}')
                proj_silhouette.start([target_floats_silhouette[ix]])

                with tqdm.trange(proj_silhouette.num_steps) as t:
                    for step in t:
                        assert step == proj_silhouette.cur_step
                        dist, loss = proj_silhouette.step()
                        t.set_postfix(dist=f'{dist[0]:.4f}', loss=f'{loss:.2f}')

                    # Save results.
                    PIL.Image.fromarray(proj_silhouette.images_uint8[0], 'RGB').save(f'{out_path_silhouette}/projsil_{image_names_silhouette[ix]}')
                    np.savez(f'{out_path_silhouette}/dlatents.npz', dlatents=proj_silhouette.dlatents)

            projection_names_silhouette = os.listdir(out_path_silhouette)
            projection_names_silhouette = [projection_name for projection_name in projection_names_silhouette if os.path.splitext(projection_name)[1] in app.config['UPLOAD_EXTENSIONS']]

        if '2' in request.form.getlist('checkbox_type'):
            
            colored_images = pix2pix.generate_images(model_path, input_images_path)
            out_pix_path = './static/pix'
            print('generated')
            for ix, colored_image in enumerate(colored_images):
                colored_image.save(os.path.join(out_pix_path,f'pix_{ix}'))

            colored_names = os.listdir(out_pix_path)

       
        return render_template('character.html', names_colored = colored_names, names_silhouette = projection_names_silhouette)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    
    return response
