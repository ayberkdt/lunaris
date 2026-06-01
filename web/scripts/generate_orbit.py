import json
import math

def get_point(r, theta, tilt_angle, phase_offset):
    # Apply phase offset to theta
    theta_total = theta + phase_offset
    x = round(r * math.cos(theta_total), 4)
    y = round(r * math.sin(theta_total) * math.sin(tilt_angle), 4)
    z = round(r * math.sin(theta_total) * math.cos(tilt_angle), 4)
    return [x, y, z]

def generate_satellite_data(r_inner, r_outer, tilt_angle, phase_offset):
    path = []
    future_paths = {}
    
    burn_frames = 30
    coast_frames = 270 
    num_frames = 1200
    
    current_theta = 0.0
    params = []
    
    burn_1_theta_p = 0.0
    burn_2_theta_p = 0.0
    burn_3_theta_p = 0.0
    burn_4_theta_p = 0.0
    
    # Calculate exact delta thetas needed for continuity
    # Phase 0: Burn 1 from nu=0 to nu=dth_b1
    dth_b1 = 30 * (2.0 + 1.5)/2.0 * math.pi / 300 # 0.175 pi
    # Phase 1: Coast 1 from nu=dth_b1 to nu=pi. Needs pi - 0.175 pi = 0.825 pi
    speed_coast_1 = (math.pi - dth_b1) / 270
    
    # Phase 1: Burn 2 from nu=pi to nu=pi + dth_b2
    dth_b2 = 30 * (1.5 + 1.0)/2.0 * math.pi / 300 # 0.125 pi
    # Phase 2: Coast 2 from nu=pi+dth_b2 to nu=2pi (or 0). Needs pi - 0.125 pi = 0.875 pi
    speed_coast_2 = (math.pi - dth_b2) / 270

    # Phase 2: Burn 3 from nu=0 to nu=dth_b3
    dth_b3 = 30 * (1.0 + 1.5)/2.0 * math.pi / 300 # 0.125 pi
    # Phase 3: Coast 3 from nu=dth_b3 to nu=pi. Needs pi - 0.125 pi = 0.875 pi
    speed_coast_3 = (math.pi - dth_b3) / 270
    
    # Phase 3: Burn 4 from nu=pi to nu=pi + dth_b4
    dth_b4 = 30 * (1.5 + 2.0)/2.0 * math.pi / 300 # 0.175 pi
    # Phase 0: Coast 0 from nu=pi+dth_b4 to nu=2pi. Needs pi - 0.175 pi = 0.825 pi
    speed_coast_0 = (math.pi - dth_b4) / 270

    for i in range(num_frames):
        phase_idx = i // 300
        frame_in_phase = i % 300
        
        if phase_idx == 0:
            if frame_in_phase < coast_frames:
                a, e, theta_p = r_inner, 0.0, current_theta - (math.pi + dth_b4 + frame_in_phase * speed_coast_0) # Dummy for circular
                theta_p = 0.0
                speed = speed_coast_0
            else:
                if frame_in_phase == coast_frames: burn_1_theta_p = current_theta
                t = (frame_in_phase - coast_frames) / burn_frames
                r_opp = r_inner * (1-t) + r_outer * t
                a, e = (r_inner + r_opp) / 2, (r_opp - r_inner) / (r_opp + r_inner)
                theta_p = burn_1_theta_p
                speed = (2.0 * (1-t) + 1.5 * t) * math.pi / 300
                
        elif phase_idx == 1:
            if frame_in_phase < coast_frames:
                a = (r_inner + r_outer) / 2
                e = (r_outer - r_inner) / (r_outer + r_inner)
                theta_p = burn_1_theta_p
                speed = speed_coast_1
            else:
                if frame_in_phase == coast_frames: burn_2_theta_p = current_theta - math.pi
                t = (frame_in_phase - coast_frames) / burn_frames
                r_peri = r_inner * (1-t) + r_outer * t
                a, e = (r_peri + r_outer) / 2, (r_outer - r_peri) / (r_outer + r_peri)
                theta_p = burn_2_theta_p
                speed = (1.5 * (1-t) + 1.0 * t) * math.pi / 300
                
        elif phase_idx == 2:
            if frame_in_phase < coast_frames:
                a, e, theta_p = r_outer, 0.0, 0.0
                speed = speed_coast_2
            else:
                if frame_in_phase == coast_frames: burn_3_theta_p = current_theta - math.pi
                t = (frame_in_phase - coast_frames) / burn_frames
                r_peri = r_outer * (1-t) + r_inner * t
                a, e = (r_peri + r_outer) / 2, (r_outer - r_peri) / (r_outer + r_peri)
                theta_p = burn_3_theta_p
                speed = (1.0 * (1-t) + 1.5 * t) * math.pi / 300
                
        elif phase_idx == 3:
            if frame_in_phase < coast_frames:
                a = (r_inner + r_outer) / 2
                e = (r_outer - r_inner) / (r_outer + r_inner)
                theta_p = burn_3_theta_p
                speed = speed_coast_3
            else:
                if frame_in_phase == coast_frames: burn_4_theta_p = current_theta
                t = (frame_in_phase - coast_frames) / burn_frames
                r_apo = r_outer * (1-t) + r_inner * t
                a, e = (r_inner + r_apo) / 2, (r_apo - r_inner) / (r_apo + r_inner)
                theta_p = burn_4_theta_p
                speed = (1.5 * (1-t) + 2.0 * t) * math.pi / 300

        nu = current_theta - theta_p
        # Avoid division by zero and maintain circle
        if e < 0.0001:
            r = a
        else:
            r = a * (1 - e**2) / (1 + e * math.cos(nu))
        
        path.append(get_point(r, current_theta, tilt_angle, phase_offset))
        params.append((a, e, theta_p, current_theta))
        
        current_theta += speed
        
    for i in range(num_frames):
        a, e, theta_p, start_theta = params[i]
        future = []
        for j in range(150):
            th = start_theta + j * (2 * math.pi / 150)
            nu = th - theta_p
            r = a if e < 0.0001 else a * (1 - e**2) / (1 + e * math.cos(nu))
            future.append(get_point(r, th, tilt_angle, phase_offset))
        future_paths[i] = future
        
    return path, future_paths

def main():
    print("Generating Keplerian Constellation Orbits...")
    
    # Premium Single Satellite (Gateway - 25 deg tilt)
    p1, f1 = generate_satellite_data(1.5, 2.4, math.radians(25), 0)
    
    output = {
        'path1': {'path': p1, 'future_paths': f1}
    }
    
    with open('../public/orbit-data.json', 'w') as f:
        json.dump(output, f)
    
    print("Orbits perfectly generated and anchored.")

if __name__ == '__main__':
    main()
