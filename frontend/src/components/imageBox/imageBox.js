import React, { useState } from "react";
import ToggleSwitch from "../toggleSwitch/ToggleSwitch";
import "./imageBox.css";

const ImageBox = ({ src, srcEncoded, alt, canEncode }) => {
  const [toggled, setToggled] = useState(false);

  const onToggle = () => {
    setToggled(!toggled);
  };

  return (
    <div className="image-box">
      {canEncode && <ToggleSwitch toggled={toggled} onToggle={onToggle} />}

      {toggled ? <img src={srcEncoded} alt={alt} /> : <img src={src} alt={alt} />}
    </div>
  );
};

export default ImageBox;
