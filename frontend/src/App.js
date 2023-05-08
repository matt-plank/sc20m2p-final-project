import "./app.css";
import InputImageBox from "./components/inputImageBox/InputImageBox";

function App() {
  return (
    <div className="image-wrapper">
      <InputImageBox canEncode={true} />
      <InputImageBox canEncode={true} />
      <button className="btn">Combine</button>
      <InputImageBox />
    </div>
  );
}

export default App;
