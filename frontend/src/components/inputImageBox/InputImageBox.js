import React, { useState } from "react";
import ToggleSwitch from "../toggleSwitch/ToggleSwitch";
import "./inputImageBox.css";

const InputImageBox = ({ src, srcEncoded, alt, canEncode }) => {
  const [toggled, setToggled] = useState(false);

  const onToggle = () => {
    setToggled(!toggled);
  };

  return (
    <div className="image-box">
      <ToggleSwitch toggled={toggled} onToggle={onToggle} />

      <button className="btn">Upload</button>

      {toggled ? <img src={srcEncoded} alt={alt} /> : <img src={src} alt={alt} />}
    </div>
  );
};

export default InputImageBox;
