import pybullet as p
import pybullet_data
import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from PPO import PPOAgent

class DriverAssistanceSim:
    def __init__(self):
        self.timeStep = 0.1
        self.num_steps = 500
        self.max_range = 3
        self.z_offset = 0.05
        self.shelf_scale = 0.5
        self.num_cars = 10
        self.step_size = 0.1 
        self.lane_width = 2.0
        self.offset = 0.1
        self.lane_width_total = 2 * self.lane_width + self.offset
        self.ang_vel = 6
        self.initialize_environment()
    
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Ground
        self.plane_id = p.loadURDF("plane.urdf")

        self.initialize_road()

        # Reset robot to original position
        self.initialize_robot()

        # Respawn cars
        self.cars = []
        self.spawn_cars(self.num_cars)
        self.trajectory = []
        self.danger_points = []

    def initialize_environment(self):
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Suppress warnings
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.setPhysicsEngineParameter(enableConeFriction=0)


    def initialize_road(self):
        self.road_id = p.loadURDF("plane.urdf")
        self.left_boundary  = self.lane_width - (self.offset * 6)
        self.right_boundary = -self.lane_width + (self.offset * 6)

        p.changeVisualShape(self.road_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1]) 
        # Draw lane lines
        p.addUserDebugLine([0, self.lane_width - self.offset, 0.01], [200, self.lane_width - self.offset, 0.01], lineColorRGB=[1, 1, 1], lineWidth=4)
        p.addUserDebugLine([0, -self.lane_width + self.offset, 0.01], [200, -self.lane_width + self.offset, 0.01], lineColorRGB=[1, 1, 1], lineWidth=4)

        lane_length = 200
        lane_color = [0, 0, 0, 1]  

        lane_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01], rgbaColor=lane_color)
        lane_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01])
        lane_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=lane_collision, baseVisualShapeIndex=lane_visual, basePosition=[lane_length/2, 0, 0])

    def initialize_robot(self):
        y = np.random.uniform(-self.lane_width + (self.offset * 5), self.lane_width - (self.offset * 5))
        self.robot_start_pos = [0, y, 0.05]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot_path = os.path.join(pybullet_data.getDataPath(), "husky/husky.urdf")
        self.robot_id = p.loadURDF(robot_path, self.robot_start_pos, self.robot_start_orientation, globalScaling=1.0)  
        self.wheel_joints = [2, 3, 4, 5] 
    
    def disconnect_environmnent(self):
        p.disconnect()

    def set_robot_wheel_velocities(self, linear_vel, angular_vel):
        WHEEL_DISTANCE = 0.55   # metres (left-right wheel separation)
        MAX_FORCE = 200

        # Convert cmd_vel → wheel velocities
        left_vel  = linear_vel - (angular_vel * WHEEL_DISTANCE / 2)
        right_vel = linear_vel + (angular_vel * WHEEL_DISTANCE / 2)

        for i, joint in enumerate(self.wheel_joints):
            if i % 2 == 0:   # left wheels = 0,2
                vel = left_vel
            else:            # right wheels = 1,3
                vel = right_vel

            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=vel,
                force=MAX_FORCE
            )

    def initialize_car(self, new_car=False):
        min_dist_between_cars = 1.0   # minimum distance between cars
        robot_safe_dist = 5.0         # minimum distance from robot start

        placed = False
        while not placed:
            x = np.random.uniform(0, 10)
            y = np.random.uniform(-self.lane_width + (self.offset * 2), self.lane_width - (self.offset * 2))
            
            # Check distance to robot
            dist_to_robot = np.sqrt((x - self.robot_start_pos[0])**2 + (y - self.robot_start_pos[1])**2)
            if dist_to_robot < robot_safe_dist:
                continue  # too close to robot, try again

            # Check distance to other cars
            overlap = False
            for car in self.cars:
                dist = np.sqrt((x - car['x_pos'])**2 + (y - car['y_pos'])**2)
                if dist < min_dist_between_cars:
                    overlap = True
                    break
            
            if not overlap:
                car_id = p.loadURDF("racecar/racecar.urdf", [x, y, 0],
                                    p.getQuaternionFromEuler([0,0,0]),
                                    globalScaling=1)
                velocity = np.random.uniform(-0.02, -0.08)
                self.cars.append({'id': car_id, 'vel': velocity, 'x_pos': x, 'y_pos': y})
                placed = True

    def clear_debug_lines(self):
        # clear old debug lines
        prev_ids = globals().get('debug_line_ids', [])
        for lid in prev_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        return []
    
    def get_angles(self, r_pos, z_offset, max_range ):
        angles = np.linspace(-math.pi/5, math.pi/5, 30, endpoint=False)
        from_points = []
        to_points = []
        for a in angles:
            ang_w = a
            start = [r_pos[0], r_pos[1], r_pos[2] ]
            end = [r_pos[0] + max_range * math.cos(ang_w),
                r_pos[1] + max_range * math.sin(ang_w),
                r_pos[2]]
            from_points.append(start)
            to_points.append(end)
        return angles, from_points, to_points
    
    def get_closest_hit(self, hits_cube):
        # return the closest hit
        if len(hits_cube) > 0:
            hits_cube.sort(key=lambda x: x[0])  # sort by distance
            closest_hit = hits_cube[0][1]
            angle = hits_cube[0][2]
            return closest_hit[0], closest_hit[1], angle
        else:
            return 0, 0, 0  # no hit
    
    def calculate_euclidean_dist(self, hit_position, r_pos, z_offset):
        return math.sqrt(
            (hit_position[0] - r_pos[0])**2 +
            (hit_position[1] - r_pos[1])**2 +
            (hit_position[2] - (r_pos[2] + z_offset))**2
        )
        
    def move_cars(self):
        for car in self.cars:
            cid = car['id']
            pos, orn = p.getBasePositionAndOrientation(cid)
            new_pos = [pos[0] - car['vel'], pos[1], pos[2]]  # move along -x
            p.resetBasePositionAndOrientation(cid, new_pos, orn)
   
    def get_lane_camera_image(self, robot_id, z_offset=0.2, forward_offset=0.3):
        base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
        local_camera_pos = [0.7, 0.0, 0.5]
        pitch_angle = -math.pi / 2
        local_camera_orn = p.getQuaternionFromEuler([0, pitch_angle, 0])
        cam_pos, cam_orn = p.multiplyTransforms(
            base_pos, base_orn, 
            local_camera_pos, local_camera_orn
        )

        cam_yaw = p.getEulerFromQuaternion(cam_orn)[2]
        forward_x = math.cos(cam_yaw)
        forward_y = math.sin(cam_yaw)
        
        target = [
            cam_pos[0],
            cam_pos[1],
            base_pos[2]  # Ground level
        ]
        
        # Up direction in world
        up_vec = [forward_x, forward_y, 0]

        view = p.computeViewMatrix(
            cameraEyePosition=cam_pos, 
            cameraTargetPosition=target, 
            cameraUpVector=up_vec
        )        
        proj = p.computeProjectionMatrixFOV(
            fov=30,
            aspect=1.0,
            nearVal=0.01,
            farVal=1.0
        )

        width, height, rgb, depth, seg = p.getCameraImage(
            width=64, height=64,
            viewMatrix=view,
            projectionMatrix=proj
        )

        rgb_array = np.array(rgb, dtype=np.uint8).reshape((height, width, 4))  # RGBA
        gray = np.dot(rgb_array[:, :, :3], [0.2989, 0.5870, 0.1140])
        gray = gray.astype(np.uint8)

        return gray, depth, seg

    def check_lane_crossing(self, robot_id, threshold= 100,forward_offset=0.3, z_offset=0.05, lateral_gap=0.1):
        rgb, _, _ = self.get_lane_camera_image(robot_id)   
        lane_visible = (np.mean(rgb) > threshold)  
        
        pos, _ = p.getBasePositionAndOrientation(robot_id)
        robot_y = pos[1]

        left_lane = False
        right_lane = False

        if lane_visible:
            if robot_y > 0:
                print("Left lane crossed")
                left_lane = True
            elif robot_y < 0:
                print("Right lane crossed")
                right_lane = True
        return left_lane, right_lane

    def show_camera_image(self, rgb):
        img = np.reshape(rgb, (128, 128, 4))[:, :, :3]
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def four_parallel_robot_beams(self, robot_pos, max_range=None, rays_per_beam=5, beam_width=0.2):
        # Emit multiple parallel rays for each of the 4 main beams (center-left, center-right, angled-left, angled-right)
        if max_range is None:
            max_range = self.max_range / 1.5

        BEAM_Z = self.z_offset
        ANG = math.radians(15)  # angled beams
        LATERAL_GAP = beam_width / 2  # spread across parallel rays
        FRONT_OFFSET = 0.3 
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        # Heading vectors
        fx = math.cos(yaw)
        fy = math.sin(yaw)
        fx_L = math.cos(yaw + ANG)
        fy_L = math.sin(yaw + ANG)
        fx_R = math.cos(yaw - ANG)
        fy_R = math.sin(yaw - ANG)

        def generate_rays(base_pos, forward_vec):
            rays_start = []
            rays_end = []
            for i in range(rays_per_beam):
                offset = -LATERAL_GAP + i * (beam_width / max(rays_per_beam-1,1))
                lat_vec = [-forward_vec[1], forward_vec[0]]
                start = [
                    base_pos[0] + offset * lat_vec[0],
                    base_pos[1] + offset * lat_vec[1],
                    base_pos[2]
                ]
                end = [
                    start[0] + forward_vec[0] * max_range,
                    start[1] + forward_vec[1] * max_range,
                    start[2]
                ]
                rays_start.append(start)
                rays_end.append(end)
            return rays_start, rays_end

        # Define main beams
        px = -math.sin(yaw)
        py = math.cos(yaw)
        front_pos = [
            robot_pos[0] + FRONT_OFFSET * fx,
            robot_pos[1] + FRONT_OFFSET * fy,
            self.z_offset
        ]

        left_base  = [robot_pos[0] + px * (0.17), robot_pos[1] + py * (0.17), BEAM_Z]  # center-left
        right_base = [robot_pos[0] - px * (0.17),  robot_pos[1] - py * (0.17),  BEAM_Z]  # center-right
        # center_base = [robot_pos[0], robot_pos[1], BEAM_Z]
        center_base = front_pos

        beams = [
            (left_base, [fx, fy]),        # straight-left
            (right_base, [fx, fy]),       # straight-right
            (center_base, [fx_L, fy_L]),  # angled-left
            (center_base, [fx_R, fy_R])   # angled-right
        ]

        # Generate all rays
        all_starts = []
        all_ends = []
        beam_ranges = []  
        for b_start, b_vec in beams:
            starts, ends = generate_rays(b_start, b_vec)
            all_starts.extend(starts)
            all_ends.extend(ends)
            beam_ranges.append(len(starts))

        results = p.rayTestBatch(all_starts, all_ends)
        car_ids = [c['id'] for c in self.cars]

        distances = []
        debug_ids = self.clear_debug_lines()
        idx = 0
        for n_rays in beam_ranges:
            dists = []
            for _ in range(n_rays):
                res = results[idx]
                frac = res[2]
                hit = res[0]
                dist = frac * max_range if hit in car_ids else max_range
                dists.append(dist)
                endp = [
                    all_starts[idx][0] + (all_ends[idx][0] - all_starts[idx][0]) * frac,
                    all_starts[idx][1] + (all_ends[idx][1] - all_starts[idx][1]) * frac,
                    all_starts[idx][2]
                ]

                if hit in car_ids:
                    # print("hit car:", hit)
                    color = [1,0,1] 
                    lid = p.addUserDebugLine(all_starts[idx], endp, color, 1, 0)
                    debug_ids.append(lid)
                
                idx += 1
            # average distance across parallel rays
            distances.append(sum(dists)/len(dists))

        globals()['debug_line_ids'] = debug_ids
        return distances[0], distances[1], distances[2], distances[3]

    def beam_sensor(self, robot_pos, z_offset=0.1, max_range=5):
        # r_pos = [robot_pos[0], robot_pos[1], z_offset]
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        fx = math.cos(yaw)
        fy = math.sin(yaw)
        forward_offset = 0.35  
        r_pos = [
            robot_pos[0] + fx * forward_offset,
            robot_pos[1] + fy * forward_offset,
            robot_pos[2] + z_offset
        ]


        new_debug_ids = self.clear_debug_lines()
        angles, from_points, to_points = self.get_angles(r_pos, z_offset, max_range)

        results = p.rayTestBatch(from_points, to_points)
        car_ids = [car['id'] for car in self.cars]
        hits_cube = []
        for i, res in enumerate(results):
            hit_object_uid = res[0] # object unique id of the hit object
            hit_fraction = res[2] # hit fraction along the ray in range [0,1] along the ray
            hit_position = res[3] # hit position in Cartesian world coordinates

            if hit_object_uid in car_ids:
                color = [1,0,0]
                ang_w = angles[i]

                measured = hit_fraction * max_range
                ep = [r_pos[0] + measured * math.cos(ang_w),
                    r_pos[1] + measured * math.sin(ang_w),
                    r_pos[2] + z_offset]
            
                lid = p.addUserDebugLine(from_points[i], ep, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
                new_debug_ids.append(lid)

                dist = self.calculate_euclidean_dist(hit_position, r_pos, z_offset)

                # print("hit uid = ", hit_object_uid, " at fraction ", hit_fraction, " at pos ", hit_position, " dist ", dist)

                hits_cube.append((dist, hit_position, ang_w))
            else:
                color = [0,1,0]
                # lid = p.addUserDebugLine(from_points[i], to_points[i], lineColorRGB=color, lineWidth=1.0, lifeTime=0)
                # new_debug_ids.append(lid)

            globals()['debug_line_ids'] = new_debug_ids
        return self.get_closest_hit(hits_cube)

    def norm_beam(self, d):
        d = np.clip(d, 0, self.max_range)
        return (self.max_range - d) / self.max_range   # 0 = safe, 1 = close

    def get_state(self, robot_pos):

        b1_left, b2_right, b3_angL, b4_angR = self.four_parallel_robot_beams(robot_pos)

        b1 = self.norm_beam(b1_left)
        b2 = self.norm_beam(b2_right)
        b3 = self.norm_beam(b3_angL)
        b4 = self.norm_beam(b4_angR)

        # Beam-based obstacle detection
        hit_x, hit_y, angle = self.beam_sensor(robot_pos, z_offset=self.z_offset, max_range=self.max_range)
        if hit_x == 0 and hit_y == 0:
            beam_dist = self.max_range
        else:
            beam_dist = np.linalg.norm(np.array([hit_x, hit_y]) - np.array(robot_pos[:2]))
        beam_dist = np.clip(beam_dist, 0, self.max_range)
        beam_dist = (self.max_range - beam_dist) / self.max_range

        # Camera clarity scores
        SIDE_THRESHOLD = 0.4

        dist_left  = abs(robot_pos[1] - self.left_boundary)
        dist_right = abs(robot_pos[1] - self.right_boundary)

        if dist_left < SIDE_THRESHOLD or dist_right < SIDE_THRESHOLD:
            left_lane, right_lane = self.check_lane_crossing(self.robot_id)
        else:
            left_lane, right_lane = 0,0

        # Lane offset
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lane_offset = np.clip(pos[1] / (self.lane_width / 2), -1.0, 1.0)    # normalized -1 to +1

        # Heading error (yaw)
        yaw = p.getEulerFromQuaternion(orn)[2]
        heading_error = np.clip(yaw / np.pi, -1.0, 1.0)

        return np.array([b1,b2,b3,b4, 
                        beam_dist, 
                        left_lane, right_lane, 
                        lane_offset, 
                        heading_error
                    ], dtype=np.float32)


    def compute_reward(self, state, action, done):
        # unpack state
        reward = 0
        b1, b2, b3, b4 = state[0], state[1], state[2], state[3]
        beam_dist      = state[4]        # front safety 0–1
        left_lane     = state[5]
        right_lane    = state[6]
        lane_offset    = abs(state[7])
        heading_error  = abs(state[8])
        
        # Obstacle avoidance using beams
        max_beam = max(b1, b2, b3, b4)
        reward += (1 - max_beam)*3     

        if max_beam > 0.85:
            reward -= 12

        if left_lane or right_lane:  # crossing lane
            reward -= 8.0

        # To encourage straight driving
        reward += 2.0 * (1.0 - lane_offset)
        reward += 0.5 * (1.0 - heading_error)

        left_clear = 1 - max(b1,b3)
        right_clear = 1 - max(b2,b4)
        SAFE = max_beam < 0.30
        DANGER = max_beam > 0.45

        if action == 0: # STRAIGHT
            if SAFE:
                reward += 1.5 # encourage straight if safe
            elif max_beam > 0.45:
                reward -= 4.0  # negative reward if going straight when it's unsafe

        elif action in (1, 2): # left/right
            if SAFE:
                reward -= 0.8 # negative reward for unnecessary turn
            else:
                # reward a turn only if it moves toward a clearer side
                left_clear = 1 - max(b1, b3)
                right_clear = 1 - max(b2, b4)
                if action == 1 and left_clear > right_clear + 0.15:
                    reward += 2.0
                if action == 2 and right_clear > left_clear + 0.15:
                    reward += 2.0
        elif action == 3:  # slow
            if max_beam > 0.55:
                reward += 1.0
            else:
                reward -= 0.5
        if done:
            reward -= 20 # collision or lane exit
        return reward

    def spawn_cars(self, car_count):
        for _ in range(car_count):
            self.initialize_car(new_car=True)
    
    def plot_metrics(self, episode_metrics, up):
        ep = episode_metrics["episode"]
        steps = episode_metrics["step"]

        fig, axs = plt.subplots(2, 2, figsize=(12,8))
        ax = axs[0,0]
        axs[0,0].plot(ep, episode_metrics["actor_loss"], label="Actor Loss", color='blue')
        axs[0,0].set_title("Actor Loss over Episodes")
        axs[0,0].set_xlabel("Episode")
        axs[0,0].set_ylabel("Loss")
        axs[0,0].grid(True)

        ax2 = ax.twinx()
        ax2.plot(ep, steps, linestyle='dotted', color='black', alpha=0.6, label="Steps")
        ax2.legend(loc="upper right")

        ax = axs[0,1]
        axs[0,1].plot(ep, episode_metrics["critic_loss"], label="Critic Loss", color='red')
        axs[0,1].set_title("Critic Loss over Episodes")
        axs[0,1].set_xlabel("Episode")
        axs[0,1].set_ylabel("Loss")
        axs[0,1].grid(True)
        ax2 = ax.twinx()
        ax2.plot(ep, steps, linestyle='dotted', color='black', alpha=0.6, label="Steps")
        ax2.legend(loc="upper right")

        ax = axs[1,0]
        axs[1,0].plot(ep, episode_metrics["entropy"], label="Entropy", color='green')
        axs[1,0].set_title("Policy Entropy over Episodes")
        axs[1,0].set_xlabel("Episode")
        axs[1,0].set_ylabel("Entropy")
        axs[1,0].grid(True)
        ax2 = ax.twinx()
        ax2.plot(ep, steps, linestyle='dotted', color='black', alpha=0.6, label="Steps")
        ax2.legend(loc="upper right")

        if episode_metrics["reward"]:
            ax = axs[1,1]
            axs[1,1].plot(ep, episode_metrics["reward"], label="Reward", color='purple')
            axs[1,1].set_title("Cumulative Reward over Episodes")
            axs[1,1].set_xlabel("Episode")
            axs[1,1].set_ylabel("Reward")
            axs[1,1].grid(True)
            ax2 = ax.twinx()
            ax2.plot(ep, steps, linestyle='dotted', color='black', alpha=0.6, label="Steps")
            ax2.legend(loc="upper right")

        plt.tight_layout()
        filename = f"metrics_{up}.png"
        plt.savefig(filename, dpi=200)
    
    def plot_trajectory(self, ep, steps, reward, loss, actor_loss, critic_loss, entropy):
        plt.figure(figsize=(10, 4))
        traj = np.array([(x, y) for (x, y, a) in self.trajectory])
        plt.plot(traj[:, 0], traj[:, 1], label="Full Trajectory")
        colors = {0: "blue", 1: "green", 2: "red", 3: "orange"}
        labels = {0: "Straight", 1: "Left", 2: "Right", 3: "Slow/Stop"}

        for x, y, a in self.danger_points:
            plt.scatter(x, y, color=colors[a], s=60, marker='X', edgecolors='black')
        plt.axhline(self.lane_width, color='black', linestyle='--')
        plt.axhline(-self.lane_width, color='black', linestyle='--')

        plt.title(f"Robot Trajectory with Actions – Episode: {ep}, Steps: {steps}, Reward: {reward:.2f}, Loss: {loss:.4f} \n Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Entropy: {entropy.item():.4f}")

        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        for a in [0, 1, 2, 3]:
            plt.scatter([], [], c=colors[a], label=labels[a])

        plt.legend()
        filename = f"trajectory_episode_{ep}.png"
        plt.savefig(filename, dpi=200)
        plt.close()

    def run(self, agent, EPISODES, is_training=True):
        reward_list = []
        initial_entropy = 0.5
        final_entropy = 0.01

        episode_metrics = {
            "episode": [],
            "step": [],
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "reward": []
            }

        for ep in range(EPISODES):
            # print("\n==== EPISODE", ep, "====")
            episode_reward = 0
            self.reset()
            terminal_step = self.num_steps
            reset_angle = 0
            self.entropy_coef = initial_entropy * (1 - ep / EPISODES) + final_entropy * (ep / EPISODES)

            for step in range(self.num_steps):
                pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                robot_pos = [pos[0], pos[1], pos[2]]

                # Step 1. Get state from environment
                state = self.get_state(robot_pos)

                # Step 2. Get action from PPO agent, by passing state to policy network
                action, logprob = agent.act(state)

                self.trajectory.append((robot_pos[0], robot_pos[1], action))
                beam = state[4]
                if beam > 0.6:
                    self.danger_points.append((robot_pos[0], robot_pos[1], action))

                # Map actions to wheel speeds
                if action == 0: # go straight 
                    self.set_robot_wheel_velocities(4, reset_angle)
                    reset_angle = 0  # reset angle for straight
                elif action == 1: # turn left
                    self.set_robot_wheel_velocities(3, +self.ang_vel)
                    reset_angle -= self.ang_vel
                elif action == 2: # turn right
                    self.set_robot_wheel_velocities(3, -self.ang_vel)
                    reset_angle += self.ang_vel
                elif action == 3: # SLOW/STOP
                    self.set_robot_wheel_velocities(1.5, reset_angle) 
                    reset_angle = 0

                p.stepSimulation()
                # time.sleep(self.timeStep)
                self.move_cars()

                new_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                new_robot_pos = [new_pos[0], new_pos[1], new_pos[2]]
                new_state = self.get_state(new_robot_pos)
                progress = new_robot_pos[0] - robot_pos[0]   # if robot moves forward, this is positive
                progress_reward = 2.0 * progress             # scale as needed

                # Terminal condition
                left_lane, right_lane = new_state[5], new_state[6]
                contacts = p.getContactPoints(bodyA=self.robot_id)
                car_ids = {car['id'] for car in self.cars}
                collision = any((c[1] in car_ids) or (c[2] in car_ids) for c in contacts)

                done = (collision or 
                        (abs(new_robot_pos[1]) > (self.lane_width + self.offset)) or 
                        (left_lane) or (right_lane))

                # Step 3. Compute reward
                reward = self.compute_reward(new_state, action,done) + progress_reward
                episode_reward += reward

                print("Episode = ",ep , "Step =", step, " State =", state, " Action =", action, " LogProb =", logprob, " Reward =", reward, "total_reward =", episode_reward)

                # Step 4. Store (s, a, logprob, r, done) in PPO memory
                agent.remember(state, action, logprob, reward, done)

                if done:
                    agent.remember(state, action, logprob, -20.0, True)
                    terminal_step = step
                    print("Episode", ep, " terminated early reward = ", episode_reward, " at step ", step)
                    break

                p.resetDebugVisualizerCamera(
                    cameraDistance=5,
                    cameraYaw=0,
                    cameraPitch=-80,
                    cameraTargetPosition=new_robot_pos  # camera follows robot
                )

            reward_list.append(episode_reward)
            print("Episode Reward =", episode_reward)

            # Step 5. Collect the above batch data, and train PPO agent after each episode using value function network
            def to_float(x):
                if torch.is_tensor(x):
                    return float(x.detach().cpu())
                return float(x)
            if( is_training):
                loss, actor_loss, critic_loss, entropy = agent.train(force=True,entropy_coef=self.entropy_coef )
                episode_metrics["episode"].append(ep)
                episode_metrics["actor_loss"].append(to_float(actor_loss))
                episode_metrics["critic_loss"].append(to_float(critic_loss))
                episode_metrics["entropy"].append(to_float(entropy))
                episode_metrics["reward"].append(to_float(episode_reward))
                episode_metrics["step"].append(terminal_step)
                self.plot_trajectory(ep, terminal_step, episode_reward, loss, actor_loss, critic_loss, entropy)
            else:
                print(f"TEST Episode {ep+1}/{EPISODES} | Total Reward: {episode_reward:.2f} | Steps: {step}")

            if (is_training and ep % 5 == 0):
                self.plot_metrics(episode_metrics, ep)
                agent.save_model("./ppo_driver_model.pth")

        print("All Episodes Completed: reward list - ", reward_list)
            
    def test_model(self, agent, total_episodes):
        print("=================Testing trained PPO agent============================")
        agent.load_model("./ppo_driver_model.pth")
        self.run(agent=agent, EPISODES=total_episodes, is_training=False )

#For training
agent = PPOAgent()
sim = DriverAssistanceSim()
EPISODES = 500
sim.run(agent, EPISODES)
sim.disconnect_environmnent()

# For testing
# sim = DriverAssistanceSim()
# agent = PPOAgent(state_dim=9, action_dim=4)
# agent.load_model("./ppo_driver_model.pth")   
# sim.test_model(agent, total_episodes=1)
# sim.disconnect_environmnent()
