import type { Meta, StoryObj } from "@storybook/react";

import BoardDetailPage from "./[id]tree";

const meta = {
  title: "PAGES/BOARD/BoardDetailPage",
  component: BoardDetailPage,
  parameters: {
    layout: "fullscreen",
  },
  tags: ["autodocs"],
} satisfies Meta<typeof BoardDetailPage>;

export default meta;

type Story = StoryObj<typeof meta>;

export const RegulationPage1: Story = {};
