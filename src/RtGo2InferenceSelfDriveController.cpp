#include <cstdlib>
#include <filesystem>
#include <iostream>

#include <cmath> 

// CNOID
#include <cnoid/MessageView>
#include <cnoid/SimpleController>
#include <cnoid/ValueTree>
#include <cnoid/YAMLReader>

// ROS
#include <geometry_msgs/Twist.h>
#include <ros/node_handle.h>

//TORCH
#include <torch/torch.h>
#include <torch/script.h>

using namespace cnoid;
namespace fs = std::filesystem;

class Go2InferenceSelfDriveController : public SimpleController
{
    // cnoid
    Body* ioBody;
    double dt;
    double inference_dt = 0.02; // genesis dt is 0.02 sec
    size_t inference_interval_steps;

    // inference
    Vector3 global_gravity;
    VectorXd last_action;
    VectorXd default_dof_pos;
    VectorXd target_dof_pos;
    std::vector<std::string> motor_dof_names;

    torch::jit::script::Module model;

    // Config values
    double P_gain;
    double D_gain;
    int num_actions;
    double action_scale;
    double ang_vel_scale;
    double lin_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    Vector3 command_scale;

    // Command resampling
    Vector3d command;
    Vector2d lin_vel_x_range;
    Vector2d lin_vel_y_range;
    Vector2d ang_vel_range;
    size_t resample_interval_steps;
    size_t step_count = 0;

    Vector3d start_pos;
    double   start_yaw;

    // Tunable gains / target
    double base_speed     = 1.5;   // [m/s]
    double K_yaw          = 2.0;   // heading ゲイン
    double K_epsi         = 0.2;   // 横ずれ→角速度フィードフォワード

public:
    virtual bool initialize(SimpleControllerIO* io) override
    {
        dt = io->timeStep();
        ioBody = io->body();

        inference_interval_steps = static_cast<int>(std::round(inference_dt / dt));
        std::ostringstream oss;
        oss << "inference_interval_steps: " << inference_interval_steps;
        MessageView::instance()->putln(oss.str());

        global_gravity = Vector3(0.0, 0.0, -1.0);

        for(auto joint : ioBody->joints()) {
            joint->setActuationMode(JointTorque);
            io->enableOutput(joint, JointTorque);
            io->enableInput(joint, JointAngle | JointVelocity);
        }
        io->enableInput(ioBody->rootLink(), LinkPosition | LinkTwist);

        // command
        command = Vector3d(0.0, 0.0, 0.0);

        // find the cfgs file
        fs::path inference_target_path = fs::path(std::getenv("HOME")) / fs::path("genesis_ws/logs/go2-walking/inference_target");
        fs::path cfgs_path = inference_target_path / fs::path("cfgs.yaml");
        if (!fs::exists(cfgs_path)) {
            oss << cfgs_path << " is not found!!!";
            MessageView::instance()->putln(oss.str());
            return false;
        }

        // load configs
        YAMLReader reader;
        auto root = reader.loadDocument(cfgs_path)->toMapping();

        auto env_cfg = root->findMapping("env_cfg");
        auto obs_cfg = root->findMapping("obs_cfg");
        auto command_cfg = root->findMapping("command_cfg");

        // env_cfg
        P_gain = env_cfg->get("kp", 20);
        D_gain = env_cfg->get("kd", 0.5);

        resample_interval_steps = static_cast<int>(std::round(env_cfg->get("resampling_time_s", 4.0) / dt));

        // base speed
        base_speed = env_cfg->get("base_speed", 1.5);
        K_yaw = env_cfg->get("K_yaw", 2.0);
        K_epsi = env_cfg->get("K_epsi", 0.2);

        // joint values
        num_actions = env_cfg->get("num_actions", 1);
        last_action = VectorXd::Zero(num_actions);
        default_dof_pos = VectorXd::Zero(num_actions);

        auto dof_names = env_cfg->findListing("dof_names");
        motor_dof_names.clear();
        for(int i=0; i<dof_names->size(); ++i){
            motor_dof_names.push_back(dof_names->at(i)->toString());
        }

        auto default_angles = env_cfg->findMapping("default_joint_angles");
        for(int i=0; i<motor_dof_names.size(); ++i){
            std::string name = motor_dof_names[i];
            default_dof_pos[i] = default_angles->get(name, 0.0);
        }

        // use default_dof_pos for initializing target angles
        target_dof_pos = default_dof_pos;

        // scales
        action_scale = env_cfg->get("action_scale", 1.0);

        // obs_cfg
        ang_vel_scale = obs_cfg->findMapping("obs_scales")->get("ang_vel", 1.0);
        lin_vel_scale = obs_cfg->findMapping("obs_scales")->get("lin_vel", 1.0);
        dof_pos_scale = obs_cfg->findMapping("obs_scales")->get("dof_pos", 1.0);
        dof_vel_scale = obs_cfg->findMapping("obs_scales")->get("dof_vel", 1.0);

        command_scale[0] = lin_vel_scale;
        command_scale[1] = lin_vel_scale;
        command_scale[2] = ang_vel_scale;

        // command_cfg
        auto range_listing = command_cfg->findListing("lin_vel_x_range");
        lin_vel_x_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());
        range_listing = command_cfg->findListing("lin_vel_y_range");
        lin_vel_y_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());
        range_listing = command_cfg->findListing("ang_vel_range");
        ang_vel_range = Vector2(range_listing->at(0)->toDouble(), range_listing->at(1)->toDouble());

        // load the network model
        fs::path model_path = inference_target_path / fs::path("policy_traced.pt");
        if (!fs::exists(model_path)) {
            oss << model_path << " is not found!!!";
            MessageView::instance()->putln(oss.str());
            return false;
        }
        model = torch::jit::load(model_path, torch::kCPU); // CPU
        model.to(torch::kCPU);
        model.eval();

        const Link* lroot = ioBody->rootLink();
        start_pos = lroot->p();
        start_yaw = std::atan2(lroot->R()(1,0), lroot->R()(0,0));

        return true;
    }


    bool inference(VectorXd& target_dof_pos, const Vector3d& angular_velocity, const Vector3d& projected_gravity, const VectorXd& joint_pos, const VectorXd& joint_vel) {
        try {
            // observation vector
            std::vector<float> obs_vec;
            for(int i=0; i<3; ++i) obs_vec.push_back(angular_velocity[i] * ang_vel_scale);
            for(int i=0; i<3; ++i) obs_vec.push_back(projected_gravity[i]);
            for(int i=0; i<3; ++i) obs_vec.push_back(command[i] * command_scale[i]);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back((joint_pos[i] - default_dof_pos[i]) * dof_pos_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(joint_vel[i] * dof_vel_scale);
            for(int i=0; i<num_actions; ++i) obs_vec.push_back(last_action[i]);

            // auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCUDA);
            auto input = torch::from_blob(obs_vec.data(), {1, (long)obs_vec.size()}, torch::kFloat32).to(torch::kCPU);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);

            // inference
            torch::Tensor output = model.forward(inputs).toTensor();
            auto output_cpu = output.to(torch::kCPU);
            auto output_acc = output_cpu.accessor<float, 2>();

            VectorXd action(num_actions);
            for(int i=0; i<num_actions; ++i){
                last_action[i] = output_acc[0][i];
                action[i] = last_action[i];
            }

            target_dof_pos = action * action_scale + default_dof_pos;
        }
        catch (const c10::Error& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
        }

        return true;
    }

    virtual bool control() override
    {   
        const auto rootLink = ioBody->rootLink();
        double x  = rootLink->p().x();
        double y  = rootLink->p().y();
        double yaw= std::atan2(rootLink->R()(1,0), rootLink->R()(0,0));
        double dx = x - start_pos.x();
        double dy = y - start_pos.y();
        double e  = -std::sin(start_yaw)*dx + std::cos(start_yaw)*dy;
        double dpsi = std::atan2(std::sin(yaw - start_yaw), std::cos(yaw - start_yaw));
        command[0] = base_speed;
        command[1] = 0.0;
        command[2] = -(K_yaw * dpsi + K_epsi * e);

        ROS_INFO("command = [%.3f, %.3f, %.3f]", command.x(), command.y(), command.z());

        // get current states
        const Isometry3d root_coord = rootLink->T();
        Vector3 angular_velocity = root_coord.linear().transpose() * rootLink->w();
        Vector3 projected_gravity = root_coord.linear().transpose() * global_gravity;

        VectorXd joint_pos(num_actions), joint_vel(num_actions);
        for(int i=0; i<num_actions; ++i){
            auto joint = ioBody->joint(motor_dof_names[i]);
            joint_pos[i] = joint->q();
            joint_vel[i] = joint->dq();
        }

        // inference
        if (step_count % inference_interval_steps == 0) {
            inference(target_dof_pos, angular_velocity, projected_gravity, joint_pos, joint_vel);
        }

        // set target outputs
        for(int i=0; i<num_actions; ++i) {
            auto joint = ioBody->joint(motor_dof_names[i]);
            double q = joint->q();
            double dq = joint->dq();
            double u = P_gain * (target_dof_pos[i] - q) + D_gain * (- dq);
            joint->u() = u;
        }

        ++step_count;

        return true;
    }
};
CNOID_IMPLEMENT_SIMPLE_CONTROLLER_FACTORY(Go2InferenceSelfDriveController)
