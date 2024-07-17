function guessProductionMode() {
    const argv = process.argv.join(" ").toLowerCase();
    const isProdEnv = process.env.NODE_ENV === "production";
    return isProdEnv || [" build", ":build", "ng b", "--prod"].some((command) => argv.includes(command));
}

process.env.TAILWIND_MODE = guessProductionMode() ? "build" : "watch";

/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./apps/**/*.{html,ts,css,scss}", "./libs/**/*.{html,ts,css,scss}"],
    darkMode: "class", // or 'media' or 'class'
    important: true,
    theme: {
        fontFamily: {
            title: ['"Nunito Sans"', '"Noto Sans"', '"Roboto"', "TimesNewRoman", '"Noto Color Emoji"'],
            content: ['"Encode Sans"', '"Segoe UI"', '"Arial"', '"Apple Color Emoji"', '"Segoe UI Emoji"', '"Segoe UI Symbol"'],
        },
    },
    variants: {
        extend: {
            backgroundColor: ["checked"],
            borderColor: ["checked"],
            inset: ["checked"],
            display: ["group-hover"],
            zIndex: ["hover", "active"],
            opacity: ({ after }) => after(["disabled"]),
        },
    },
    plugins: [require("@tailwindcss/forms"), require("@tailwindcss/typography")],
};
