import React from "react";
import "./toggleSwitch.css";

const ToggleSwitch = ({ toggled, onToggle }) => {
  return (
    <div className="toggle-container">
      <label className="switch">
        <input type="checkbox" checked={toggled} onClick={onToggle} />
        <span className="slider round"></span>
      </label>
      <p>{toggled ? "Encoded" : "Original"}</p>
    </div>
  );
};

export default ToggleSwitch;
